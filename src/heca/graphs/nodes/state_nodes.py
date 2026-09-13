import torch

from heca.graphs.nodes.node import StateNode
from heca.graphs.nodes.node_set import NodeSet
from heca.graphs.roles import ROLE_ALL


class StateNodes(NodeSet[StateNode]):
    type = "state"
    state = "state"

    FEATURE_DIM = 1

    def __init__(self):
        super().__init__()
        self.add(self.state, StateNode(role=ROLE_ALL))

    def build(self, budget: float):
        assert len(self.items) == 1
        self.x = torch.full((1, self.FEATURE_DIM), float(budget), dtype=torch.float32)
        self.type_ids = torch.tensor(
            [node.role for node in self.items], dtype=torch.long
        )
