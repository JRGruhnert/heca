import numpy as np
import torch

from heca.data.data import DCScene
from heca.data.entity import Entity
from heca.graphs.nodes.node import EntityNode
from heca.graphs.nodes.node_set import NodeSet
from heca.graphs.roles import ENMode


class EntityNodes(NodeSet[EntityNode]):
    type = "entity"

    def build(self, start: DCScene, goal: DCScene, use_rotation: bool = True):
        self.update_nodes(start, goal)
        x_np = np.stack([node.data.feature for node in self.items], axis=0)
        self.x = torch.from_numpy(x_np).float()
        if not use_rotation:
            self.x = Entity.without_rotation(self.x, Entity.POINT_LAYOUT)
        self.type_ids = torch.tensor(
            [node.type_id for node in self.items], dtype=torch.long  # type: ignore
        )
        self.role_ids = torch.tensor(
            [node.role.value for node in self.items], dtype=torch.long
        )

    def update_nodes(self, start: DCScene, goal: DCScene):
        for node in self.items:
            if node.mode == ENMode.START:
                node.data = start.get(node.entity)
            elif node.mode == ENMode.GOAL:
                node.data = goal.get(node.entity)
            elif node.mode == ENMode.SAMPLE:
                assert node.con is not None
                node.data = node.con.sample(node.entity)
