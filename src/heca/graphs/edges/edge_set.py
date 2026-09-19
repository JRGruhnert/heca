from typing import ClassVar, Generic, TypeVar

import numpy as np
import torch

from heca.data.entity import Entity
from heca.graphs.nodes.node_set import NodeSet
from heca.graphs.nodes.node import GraphNode
from heca.utils.quaternion import Quaternion

S = TypeVar("S", bound=GraphNode)
D = TypeVar("D", bound=GraphNode)


RESIDUAL_TERMS: tuple[str, ...] = ("z", "z2", "logz", "logp", "state", "rbf")
DEFAULT_TERMS: tuple[str, ...] = ("z", "rbf", "state")
# DEFAULT_TERMS: tuple[str, ...] = ("z", "z2", "logz", "logp", "state")


class EdgeSet(Generic[S, D]):
    type: ClassVar[tuple[str, str, str]]
    has_attrs: bool = True
    rho_max: float = 1.25
    rbf_centers: int = 6

    @staticmethod
    def validate_terms(terms: tuple[str, ...]) -> None:
        """Reject unknown or empty selections loudly instead of silently shrinking."""
        unknown = [term for term in terms if term not in RESIDUAL_TERMS]
        if unknown:
            raise ValueError(
                f"unknown residual term(s) {unknown}; known: {', '.join(RESIDUAL_TERMS)}"
            )
        if not terms:
            raise ValueError("the edge feature needs at least one residual term")

    def rbf_features(self, rho: np.ndarray, eps: float = 1e-15) -> np.ndarray:
        rho = np.asarray(rho, dtype=np.float64).reshape(-1, 1)
        centres = np.linspace(0.0, self.rho_max, self.rbf_centers).reshape(1, -1)
        return np.exp(
            -((rho - centres) ** 2)
            / (2.0 * (self.rho_max / (self.rbf_centers - 1)) ** 2 + eps)
        )

    @classmethod
    def _term_width(cls, term: str, use_rotation: bool) -> int:
        """Columns one residual term contributes."""
        rotation = Entity.ROT_DIM if use_rotation else 0
        if term == "rbf":
            return cls.rbf_centers
        if term == "z":
            return Entity.POS_DIM + rotation + Entity.MAX_EXTRA_DIM
        if term == "z2":
            return Entity.POS_DIM + rotation + Entity.MAX_EXTRA_DIM
        if term == "logz":
            return Entity.POS_DIM + rotation + Entity.MAX_EXTRA_DIM
        if term in ("logp", "state"):
            return 1
        raise ValueError(
            f"unknown residual term {term!r}; known: {', '.join(RESIDUAL_TERMS)}"
        )

    @classmethod
    def residual_dim(
        cls,
        terms: tuple[str, ...] = DEFAULT_TERMS,
        use_rotation: bool = True,
    ) -> int:
        cls.validate_terms(terms)
        return sum(cls._term_width(term, use_rotation) for term in terms)

    def __init__(self):
        self.edge_index: torch.Tensor = torch.empty((2, 0), dtype=torch.long)
        self.edge_attr: torch.Tensor = torch.empty((2, 0), dtype=torch.long)
        self.edges: list[tuple[int, int]] = []
        self.attrs: list[np.ndarray] = []
        self.rebuild: bool = True

    def add(self, src_idx: int, dst_idx: int):
        self.edges.append((src_idx, dst_idx))
        if self.has_attrs:
            self.attrs.append(np.zeros(0))
        self.rebuild = True

    def reset(self):
        """Drop the previous build; every edge is rebuilt from the node sets."""
        self.edges.clear()
        self.attrs.clear()
        self.rebuild = True

    @property
    def size(self) -> int:
        return len(self.edges)

    @staticmethod
    def gather_features(nset: NodeSet, indices) -> np.ndarray:
        assert nset.x.shape[0] == len(nset.items), f"{nset.type} node set not built"
        return nset.x.numpy()[np.asarray(indices, dtype=np.intp)]

    def build(self, snset: NodeSet[S], dnset: NodeSet[D]):
        self.reset()
        self.edges_from_sets(snset, dnset)
        src_list, dst_list = zip(*self.edges)
        self.edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        if not self.has_attrs:
            self.edge_attr = torch.empty((2, 0), dtype=torch.long)
            return
        for i, edge in enumerate(self.edges):
            src = snset.idx_get(edge[0])
            dst = dnset.idx_get(edge[1])
            self.update_attr(src, dst, i)
        self.edge_attr = torch.from_numpy(np.stack(self.attrs)).float()

    def update_attr(self, src: S, dst: D, index: int):
        raise NotImplementedError

    def edges_from_sets(self, snset: NodeSet[S], tnset: NodeSet[D]):
        for i, node in enumerate(tnset.items):
            for key in node.sources.get(snset.type, ()):
                if snset.has_key(key):
                    j = snset.get_index(key)
                    self.add(j, i)

    def __str__(self) -> str:
        return (
            f"EdgeSet({self.type[0]}→{self.type[2]}): "
            f"{self.size} edges, attr.shape={self.edge_attr.shape}"
        )

    def residual(
        self,
        x_src: np.ndarray,
        x_dst: np.ndarray,
        radius: np.ndarray | None = None,
    ) -> np.ndarray:
        """Edge feature for a single src/dst feature pair (``[F]`` each)."""
        return self.residual_batch(x_src, x_dst, radius=radius)[0]

    def residual_batch(
        self,
        x_src: np.ndarray,
        x_dst: np.ndarray,
        eps=1e-15,
        use_rotation: bool = True,
        terms: tuple[str, ...] = DEFAULT_TERMS,
        radius: np.ndarray | None = None,
    ) -> np.ndarray:
        EdgeSet.validate_terms(terms)
        # the layouts the graph actually exports for this rotation mode
        Lc = Entity.layout(use_rotation, logstd=True)
        Lp = Entity.layout(use_rotation, logstd=False)
        x_src = np.atleast_2d(x_src)
        x_dst = np.atleast_2d(x_dst)

        mu_pos = x_src[:, Lc["pos"].mean()]
        ls_pos = x_src[:, Lc["pos"].logstd()]
        mu_extra = x_src[:, Lc["extra"].mean()]
        ls_extra = x_src[:, Lc["extra"].logstd()]

        x_pos = x_dst[:, Lp["pos"].mean()]
        x_extra = x_dst[:, Lp["extra"].mean()]

        z_pos_raw = (x_pos - mu_pos) * np.exp(-ls_pos)
        z_extra_raw = (x_extra - mu_extra) * np.exp(-ls_extra)

        if use_rotation:
            q_c = x_src[:, Lc["rot"].mean()]
            ls_rot = x_src[:, Lc["rot"].logstd()]
            q_x = x_dst[:, Lp["rot"].mean()]
            r_vec = Quaternion.log_map(Quaternion.mul(q_x, Quaternion.inv(q_c)))
            z_rot_raw = r_vec * np.exp(-ls_rot)
            logp_rot = np.sum(
                np.log(2 * np.pi) + 2.0 * ls_rot + z_rot_raw**2, axis=-1, keepdims=True
            )
        else:
            # no rotation: the block is dropped from the feature, not zeroed
            z_rot_raw = None
            logp_rot = np.zeros((x_pos.shape[0], 1))

        logz_pos = np.log1p(np.abs(z_pos_raw))
        logz_extra = np.log1p(np.abs(z_extra_raw))

        rbf_block: np.ndarray | None = None
        if "rbf" in terms:
            if radius is None:
                raise ValueError(
                    "the 'rbf' term needs the per-edge acceptance radius "
                    "(Entity.acceptance_radius) to normalise the residual"
                )
            parts = [z_pos_raw, z_extra_raw]
            if z_rot_raw is not None:
                parts.append(z_rot_raw)
            r_raw = np.sqrt(
                sum(np.sum(part**2, axis=-1, keepdims=True) for part in parts)
            )
            r_max = np.asarray(radius, dtype=np.float64).reshape(-1, 1)
            if np.any(r_max <= 0.0):
                raise ValueError(
                    f"acceptance radius must be positive, got {r_max.min()}"
                )
            rbf_block = self.rbf_features(r_raw / r_max)

        z_pos = np.clip(z_pos_raw, -Entity.Z_CLIP, Entity.Z_CLIP)
        z_extra = np.clip(z_extra_raw, -Entity.Z_CLIP, Entity.Z_CLIP)
        if z_rot_raw is not None:
            logz_rot = np.log1p(np.abs(z_rot_raw))
            z_rot = np.clip(z_rot_raw, -Entity.Z_CLIP, Entity.Z_CLIP)

        # log-density of the value under the component (diagonal Gaussian)
        logp = -0.5 * (
            np.sum(
                np.log(2 * np.pi) + 2.0 * ls_pos + z_pos_raw**2, axis=-1, keepdims=True
            )
            + logp_rot
            + np.sum(
                np.log(2 * np.pi) + 2.0 * ls_extra + z_extra_raw**2,
                axis=-1,
                keepdims=True,
            )
        )
        logp = np.clip(logp, -Entity.LOGP_CLIP, Entity.LOGP_CLIP)

        p_c = x_src[:, Lc["state"].mean()]
        p_x = x_dst[:, Lp["state"].mean()]
        z_state = -np.sum(p_x * np.log(p_c + eps), axis=-1, keepdims=True)
        z_state = np.clip(z_state, 0.0, Entity.Z_CLIP)

        columns: list[tuple[str, np.ndarray]] = [
            ("z", z_pos),
            ("z2", z_pos**2),
            ("logz", logz_pos),
        ]
        if z_rot_raw is not None:
            columns += [
                ("z", z_rot),
                ("z2", z_rot**2),
                ("logz", logz_rot),
            ]
        columns += [
            ("z", z_extra),
            ("z2", z_extra**2),
            ("logz", logz_extra),
        ]
        if rbf_block is not None:
            columns.append(("rbf", rbf_block))
        columns += [
            ("logp", logp),
            ("state", z_state),
        ]
        return np.concatenate(
            [block for term, block in columns if term in terms], axis=-1
        )
