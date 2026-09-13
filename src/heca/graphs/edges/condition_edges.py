import numpy as np
import torch

from heca.graphs.edges.edge_set import EdgeSet
from heca.graphs.nodes.comp_nodes import CompNodes
from heca.graphs.nodes.entity_nodes import EntityNodes
from heca.graphs.nodes.node import CompNode, EntityNode
from heca.graphs.nodes.node_set import NodeSet


class ConditionEdges(EdgeSet[CompNode, EntityNode]):
    type = (CompNodes.type, "condition", EntityNodes.type)

    def build(self, snset: NodeSet[CompNode], dnset: NodeSet[EntityNode]):
        self.edges_from_sets(snset, dnset)
        src_list, dst_list = zip(*self.edges)
        self.edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

        x_src = self.gather_features(snset, src_list)
        x_dst = self.gather_features(dnset, dst_list)
        assert all(
            snset.items[i].n_states == dnset.items[j].n_states for i, j in self.edges
        ), "condition edges must share a state vocabulary"
        weights = np.fromiter(
            (snset.items[int(i)].weight for i in src_list),
            dtype=np.float32,
            count=len(src_list),
        )

        attr = np.concatenate(
            [self.residual_batch(x_src, x_dst), weights[:, None]], axis=-1
        )
        self.attrs = list(attr)
        self.edge_attr = torch.from_numpy(attr).float()
