import numpy as np
import torch

from heca.graphs.node_set import NodeSet
from heca.graphs.nodes import GraphNode
from heca.misc.data import DCScene
from heca.misc.quaternion import Quaternion


class EdgeSet:
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

    def build(self, snset: NodeSet, dnset: NodeSet):
        src_list, dst_list = zip(*self.edges)
        self.edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        for i, edge in enumerate(self.edges):
            src = snset.idx_get(edge[0])
            dst = dnset.idx_get(edge[1])
            if src.changed or dst.changed:
                self.update_attr(src, dst, i, goal)
        self.edge_attr = torch.from_numpy(self.attrs).float()

    # One after another (you got this. its just a master thesis. you wont fail)
    # TODO: implement update attr
    # implement subgoal option get
    # implement network
    # cleanup comp con parameter
    # cleanup tapas model state check
    # clean up recorded data
    # record tapas model
    # debug
    # fix ppo databuffer
    # think about what tests to perform
    # fix image encoder (uncertainty)
    # record image samples
    # recordtapas from image encoder

    def update_attr(self, src: GraphNode, dst: GraphNode, index: int, goal: DCScene):
        if self.type == ("entity", "stepmix", "entity"):
            self.attrs[index] = self.compute_edge_feats(src.data, dst.data)
        elif self.type == ("entity", "summary", "option"):
            pass
        elif self.type == ("entity", "tapas", "entity"):
            self.attrs[index] = np.empty(1)
        else:
            raise NotImplementedError

    def compute_edge_feats(self, x_src: np.ndarray, x_dst: np.ndarray) -> np.ndarray:
        """
        Compute directed edge features from source nodes to destination nodes.

        Args:
            x_src: [E, 13+K] features for source nodes (e.g., Components)
            x_dst: [E, 13+K] features for destination nodes (e.g., Entities)

        Returns:
            edge_feat: [E, 7] normalized residuals (z_pos + z_rot + z_state)
        """
        # Unpack source (A)
        mu_pos_a = x_src[:, 0:3]
        lstd_pos_a = x_src[:, 3:6]
        q_a = x_src[:, 6:10]
        lstd_rot_a = x_src[:, 10:13]
        logits_a = x_src[:, 13:]

        # Unpack destination (B)
        mu_pos_b = x_dst[:, 0:3]
        lstd_pos_b = x_dst[:, 3:6]
        q_b = x_dst[:, 6:10]
        lstd_rot_b = x_dst[:, 10:13]
        logits_b = x_dst[:, 13:]

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

        # --- 3. State Residual (Negative Log-Likelihood) ---
        # Get destination's true class index (argmax of its logits)
        c_b = np.argmax(logits_b, axis=-1)  # [E]

        # Compute softmax of source logits
        logits_exp = np.exp(logits_a - np.max(logits_a, axis=-1, keepdims=True))
        softmax_a = logits_exp / np.sum(logits_exp, axis=-1, keepdims=True)

        # Get probability of destination's class under source distribution
        prob = softmax_a[np.arange(len(c_b)), c_b]  # [E]
        z_state = -np.log(prob + 1e-8)  # [E]

        # Stack
        edge_feat = np.concatenate(
            [z_pos, z_rot, z_state[:, np.newaxis]],
            axis=-1,  # [E, 3]  # [E, 3]  # [E, 1]
        )  # [E, 7]

        return edge_feat

    def build_edges(
        self,
        comp_feats: np.ndarray,
        entity_feats: np.ndarray,
        edge_index: np.ndarray,
    ):
        """
        Build edge features for all edges in a single shot.

        Args:
            comp_features: [num_components, 13+K] node features for components
            entity_features: [num_entities, 13+K] node features for entities
            edge_index: [2, E] where edge_index[0] = src indices, edge_index[1] = dst indices
                        Indices refer to the combined node array [comp_features; entity_features]

        Returns:
            edge_attr: [E, 7] computed edge features
            edge_type: [E] integer types (0 = comp->entity, 1 = entity->comp, etc.)
        """
        # Combine all nodes
        all_nodes = np.concatenate([comp_feats, entity_feats], axis=0)
        num_comps = comp_feats.shape[0]

        src_idx = edge_index[0]
        dst_idx = edge_index[1]

        # Determine edge types based on src/dst ranges
        is_comp_to_entity = (src_idx < num_comps) & (dst_idx >= num_comps)
        is_entity_to_comp = (src_idx >= num_comps) & (dst_idx < num_comps)
        # You can also have comp->comp or entity->entity if needed

        # Pull source and destination features
        x_src = all_nodes[src_idx]  # [E, 13+K]
        x_dst = all_nodes[dst_idx]  # [E, 13+K]

        # Compute features
        edge_attr = self.compute_edge_feats(x_src, x_dst)

        # Assign edge types (0 = comp->entity, 1 = entity->comp)
        edge_type = np.zeros(len(src_idx), dtype=np.int64)
        edge_type[is_entity_to_comp] = 1

        return edge_attr, edge_type
