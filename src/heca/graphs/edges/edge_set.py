from typing import ClassVar, Generic, TypeVar

import numpy as np
import torch

from heca.data.entity import Entity
from heca.graphs.nodes.node_set import NodeSet
from heca.graphs.nodes.node import GraphNode
from heca.utils.quaternion import Quaternion

S = TypeVar("S", bound=GraphNode)
D = TypeVar("D", bound=GraphNode)


class EdgeSet(Generic[S, D]):
    type: ClassVar[tuple[str, str, str]]
    has_attrs: bool = True

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
        """Create edges by matching node source entries to this edge type.

        A row without an entry for this source type simply has no incoming edge
        of this kind (e.g. the current/goal rows are plain values), so a missing
        key is not an error.
        """
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
    ) -> np.ndarray:
        """Edge feature for a single src/dst feature pair (``[F]`` each)."""
        return self.residual_batch(x_src, x_dst)[0]

    @staticmethod
    def residual_batch(
        x_src: np.ndarray,
        x_dst: np.ndarray,
        eps=1e-15,
        use_rotation: bool = True,
    ) -> np.ndarray:
        Lc = Entity.LAYOUT
        Lp = Entity.POINT_LAYOUT
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
            z_rot_raw = np.zeros_like(x_pos)
            logp_rot = np.zeros((x_pos.shape[0], 1))

        logz_pos = np.log1p(np.abs(z_pos_raw))
        logz_rot = np.log1p(np.abs(z_rot_raw))
        logz_extra = np.log1p(np.abs(z_extra_raw))

        z_pos = np.clip(z_pos_raw, -Entity.Z_CLIP, Entity.Z_CLIP)
        z_rot = np.clip(z_rot_raw, -Entity.Z_CLIP, Entity.Z_CLIP)
        z_extra = np.clip(z_extra_raw, -Entity.Z_CLIP, Entity.Z_CLIP)

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

        # Cross-entropy between the value's one-hot state and the component's
        # posterior: -log p_component(state_x)
        p_c = x_src[:, Lc["state"].mean()]
        p_x = x_dst[:, Lp["state"].mean()]
        z_state = -np.sum(p_x * np.log(p_c + eps), axis=-1, keepdims=True)
        z_state = np.clip(z_state, 0.0, Entity.Z_CLIP)

        return np.concatenate(
            [
                z_pos,
                z_pos**2,
                logz_pos,
                z_rot,
                z_rot**2,
                logz_rot,
                z_extra,
                z_extra**2,
                logz_extra,
                logp,
                z_state,
            ],
            axis=-1,
        )
