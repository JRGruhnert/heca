import torch

from heca.graphs.edges.edge_set import EdgeSet
from heca.graphs.nodes.node import OptionNode, StateNode
from heca.graphs.nodes.node_set import NodeSet
from heca.graphs.nodes.option_nodes import OptionNodes
from heca.graphs.nodes.state_nodes import StateNodes


class SceneEdges(EdgeSet[OptionNode, StateNode]):
    type = (OptionNodes.type, "scene", StateNodes.type)
    has_attrs: bool = True

    def build(self, snset: OptionNodes, tnset: NodeSet[StateNode]):
        """One ``option -> state`` edge per option, carrying only its sign.

        +1 for a feasible (ungated) option, -1 for a gated one; the state node
        aggregates them, so no budget and no per-view nodes are needed here. The
        gate is the tensor ``OptionNodes.build`` computed (the per-node field is
        never set).
        """
        attrs: list[float] = []
        gated = snset.gated
        j = tnset.get_index(StateNodes.state)
        for i in range(len(snset.items)):
            self.add(i, j)
            attrs.append(-1.0 if bool(gated[i]) else 1.0)

        src_list, dst_list = zip(*self.edges)
        self.edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        self.edge_attr = torch.tensor(attrs, dtype=torch.float32).reshape(-1, 1)
