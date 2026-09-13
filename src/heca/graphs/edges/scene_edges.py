import torch

from heca.graphs.edges.edge_set import EdgeSet
from heca.graphs.nodes.node import OptionNode, StateNode
from heca.graphs.nodes.node_set import NodeSet
from heca.graphs.nodes.option_nodes import OptionNodes
from heca.graphs.nodes.state_nodes import StateNodes


class SceneEdges(EdgeSet[OptionNode, StateNode]):
    type = (OptionNodes.type, "scene", StateNodes.type)
    has_attrs: bool = True

    def build(
        self,
        snset: NodeSet[OptionNode],
        tnset: NodeSet[StateNode],
        budget: float,
    ):
        self.attrs: list[tuple[float, float]] = []
        for i, onode in enumerate(snset.items):
            ja = tnset.get_index(StateNodes.all)
            if onode.gated:
                jg = tnset.get_index(StateNodes.gated)
                self.add(i, jg)
                self.attrs.append((-1.0, budget))
            else:
                jn = tnset.get_index(StateNodes.ungated)
                self.add(i, jn)
                self.attrs.append((1.0, budget))
            self.add(i, ja)
            self.attrs.append((0.0, budget))

        src_list, dst_list = zip(*self.edges)
        self.edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        self.edge_attr = torch.from_numpy(self.attrs).float()
