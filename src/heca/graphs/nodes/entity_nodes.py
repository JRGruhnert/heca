import numpy as np
import torch

from heca.data.data import DCScene
from heca.graphs.nodes.node import EntityNode, ValueMode
from heca.graphs.nodes.node_set import NodeSet


class EntityNodes(NodeSet[EntityNode]):
    type = "entity"

    def build(self, start: DCScene, goal: DCScene):
        self.update_nodes(start, goal)
        x_np = np.stack([node.data.feature for node in self.items], axis=0)
        self.x = torch.from_numpy(x_np).float()
        self.type_ids = torch.tensor(
            [node.type_id for node in self.items], dtype=torch.long  # type: ignore
        )
        self.role_ids = torch.tensor(
            [node.role for node in self.items], dtype=torch.long
        )

    def update_nodes(self, start: DCScene, goal: DCScene):
        for node in self.items:
            if node.vmode == ValueMode.START:
                node.data = start.get(node.entity)
            elif node.vmode == ValueMode.GOAL:
                node.data = goal.get(node.entity)
            elif node.vmode == ValueMode.SAMPLE:
                assert node.con is not None
                node.data = node.con.sample(node.entity)
