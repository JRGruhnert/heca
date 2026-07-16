from typing import Generic, TypeVar

import numpy as np
import torch

from heca.graphs.node_set import NodeSet
from heca.graphs.nodes import EntityNode, GraphNode, OptionNode
from heca.utils.quaternion import Quaternion

S = TypeVar("S", bound=GraphNode)
D = TypeVar("D", bound=GraphNode)


class EdgeSet(Generic[S, D]):
    def __init__(self, type: tuple[str, str, str]):
        self.edge_index: torch.Tensor = torch.empty((2, 0), dtype=torch.long)
        self.edge_attr: torch.Tensor = torch.empty((2, 0), dtype=torch.long)
        self.edges: list[tuple[int, int]] = []
        self.attrs: list[np.ndarray] = []
        self.rebuild: bool = True
        self.type = type

    def add(self, src_idx: int, dst_idx: int):
        self.edges.append((src_idx, dst_idx))
        self.attrs.append(np.zeros(0))
        self.rebuild = True

    def size(self) -> int:
        return len(self.edges)

    def build(self, snset: NodeSet[S], dnset: NodeSet[D]):
        src_list, dst_list = zip(*self.edges)
        self.edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        for i, edge in enumerate(self.edges):
            src = snset.idx_get(edge[0])
            dst = dnset.idx_get(edge[1])
            if src.changed or dst.changed:
                self.update_attr(src, dst, i)
        self.edge_attr = torch.from_numpy(self.attrs).float()

    def update_attr(self, src: S, dst: D, index: int):
        if self.type == ("entity", "stepmix", "entity"):
            assert isinstance(src, EntityNode)
            self.attrs[index] = self.stepmix_feat(
                src.data.feature, dst.data.feature, src.weight
            )
        elif self.type == ("entity", "summary", "option"):
            assert isinstance(src, EntityNode)
            assert isinstance(dst, OptionNode)
            self.attrs[index] = self.summary_feat(
                src.data.feature, dst.data[src.entity].feature
            )
        elif self.type == ("entity", "tapas", "entity"):
            self.attrs[index] = np.empty(1)
        else:
            raise NotImplementedError

    def stepmix_feat(
        self, x_src: np.ndarray, x_dst: np.ndarray, w_src: float
    ) -> np.ndarray:
        feat = self.residual(x_src, x_dst)
        return np.concatenate([feat, w_src])

    def summary_feat(self, x_src: np.ndarray, x_dst: np.ndarray) -> np.ndarray:
        return self.residual(x_src, x_dst)

    def tapas_feat(self, x_src: np.ndarray, x_dst: np.ndarray) -> np.ndarray:
        return np.empty(1)

    def residual(self, x_src: np.ndarray, x_dst: np.ndarray) -> np.ndarray:
        """
        Compute directed edge features from source nodes to destination nodes.

        Args:
            x_src: [13+K] features for src nodes
            x_dst: [13+K] features for dst nodes

        Returns:
            edge_feat: [8] normalized residuals (z_pos + z_rot + z_state + w_src)
        """
        # Unpack source (A)
        mu_pos_a = x_src[0:3]
        lstd_pos_a = x_src[3:6]
        q_a = x_src[6:10]
        lstd_rot_a = x_src[10:13]
        logits_a = x_src[13:]

        # Unpack destination (B)
        mu_pos_b = x_dst[0:3]
        lstd_pos_b = x_dst[3:6]
        q_b = x_dst[6:10]
        lstd_rot_b = x_dst[10:13]
        logits_b = x_dst[13:]

        # --- 1. Position Residual ---
        var_pos_a = np.exp(2 * lstd_pos_a)
        var_pos_b = np.exp(2 * lstd_pos_b)
        var_comb_pos = var_pos_a + var_pos_b
        z_pos = (mu_pos_b - mu_pos_a) / np.sqrt(var_comb_pos + 1e-8)

        # --- 2. Rotation Residual ---
        # q_rel = q_b * q_a^{-1}
        q_inv = Quaternion.inv(q_a)
        q_rel = Quaternion.mul(q_b, q_inv)
        r_vec = Quaternion.log_map(q_rel)  # [E, 3]

        var_rot_a = np.exp(2 * lstd_rot_a)
        var_rot_b = np.exp(2 * lstd_rot_b)
        var_comb_rot = var_rot_a + var_rot_b
        z_rot = r_vec / np.sqrt(var_comb_rot + 1e-8)

        # --- 3. State Residual (Cross-Entropy) ---
        # Softmax of both distributions
        logits_a_max = np.max(logits_a, axis=-1, keepdims=True)
        logits_b_max = np.max(logits_b, axis=-1, keepdims=True)
        softmax_a = np.exp(logits_a - logits_a_max) / np.sum(
            np.exp(logits_a - logits_a_max), axis=-1, keepdims=True
        )
        softmax_b = np.exp(logits_b - logits_b_max) / np.sum(
            np.exp(logits_b - logits_b_max), axis=-1, keepdims=True
        )

        # Cross-entropy H(B, A) = -Σ P_B(c) * log P_A(c)
        # This is the expected surprise of B's full distribution under A's belief.
        z_state = -np.sum(softmax_b * np.log(softmax_a + 1e-8), axis=-1)
        # Clip to prevent extreme outliers from dominating the edge feature
        z_state = np.clip(z_state, 0.0, 10.0)

        # Stack
        edge_feat = np.concatenate(
            [z_pos, z_rot, z_state[:, np.newaxis]], axis=-1
        )  # [8]

        return edge_feat

    def edges_from_sets(self, snset: NodeSet[S], tnset: NodeSet[D]):
        """Create edges by matching node source entries to this edge type."""
        for i, node in enumerate(tnset.items):
            for e, key in node.sources:
                if e == self.type[1]:  # matches e.g. "stepmix", "summar y", "tapas"
                    if snset.has_key(key):
                        j = snset.get_index(key)
                        self.add(j, i)
