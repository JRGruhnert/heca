import torch

from heca.graphs.nodes.node import StateNode
from heca.graphs.nodes.node_set import NodeSet


class StateNodes(NodeSet[StateNode]):
    type = "state"

    FEATURE_DIM = 1

    def __init__(self):
        super().__init__()
        self.add(self.type, StateNode())

    def build(self, budget: float):
        assert len(self.items) == 1
        self.x = torch.full((1, self.FEATURE_DIM), budget, dtype=torch.float32)
        self.type_ids = torch.tensor(
            [node.role.value for node in self.items], dtype=torch.long
        )
