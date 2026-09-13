import torch

from heca.graphs.nodes.node import StateNode
from heca.graphs.nodes.node_set import NodeSet
from heca.graphs.roles import ROLE_ALL, ROLE_GATED, ROLE_UNGATED


class StateNodes(NodeSet[StateNode]):
    type = "state"
    ungated = "ungated"
    gated = "gated"
    all = "all"

    def __init__(self):
        super().__init__()

        self.add(self.ungated, StateNode(role=ROLE_UNGATED))
        self.add(self.all, StateNode(role=ROLE_ALL))
        self.add(self.gated, StateNode(role=ROLE_GATED))

    def build(self, budget: float):
        self.x = torch.tensor((len(self.items), [budget]), dtype=torch.float32)
        self.type_ids = torch.tensor(
            [node.role for node in self.items], dtype=torch.long
        )
