import numpy as np
import torch

from heca.data.entity import Entity
from heca.graphs.nodes.node import CompNode
from heca.graphs.nodes.node_set import NodeSet


class CompNodes(NodeSet[CompNode]):
    type = "comp"

    def build(self, use_rotation: bool = True):
        x_np = np.stack([node.data.feature for node in self.items], axis=0)
        self.type_ids = torch.tensor(
            [node.type_id for node in self.items], dtype=torch.long  # type: ignore
        )
        self.weights = torch.tensor(
            [node.weight for node in self.items], dtype=torch.float32
        )
        self.x = torch.from_numpy(x_np).float()
        if not use_rotation:
            self.x = Entity.project(self.x, Entity.NO_ROT_LAYOUT)
