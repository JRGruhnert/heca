import torch

from heca.graphs.edges.edge_set import EdgeSet
from heca.graphs.nodes.node import OptionNode, StateNode
from heca.graphs.nodes.node_set import NodeSet
from heca.graphs.nodes.option_nodes import OptionNodes
from heca.graphs.nodes.state_nodes import StateNodes


class SceneEdges(EdgeSet[OptionNode, StateNode]):
    type = (OptionNodes.type, "scene", StateNodes.type)
    has_attrs: bool = False

    def build(self, snset: OptionNodes, tnset: NodeSet[StateNode]):
        self.reset()
        j = tnset.get_index(StateNodes.type)
        for i in range(len(snset.items)):
            self.add(i, j)

        src_list, dst_list = zip(*self.edges)
        self.edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
