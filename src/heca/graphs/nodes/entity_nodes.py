import numpy as np
import torch

from heca.data.data import DCScene
from heca.data.entity import Entity
from heca.graphs.nodes.node import EntityNode
from heca.graphs.nodes.node_set import NodeSet
from heca.graphs.roles import ENMode


class EntityNodes(NodeSet[EntityNode]):
    type = "entity"

    def __init__(self):
        super().__init__()
        self.entity_ids: torch.Tensor = torch.empty(0, dtype=torch.long)

    def build(self, start: DCScene, goal: DCScene, use_rotation: bool):
        self.update_nodes(start, goal)
        x_np = np.stack([node.data.feature for node in self.items], axis=0)
        self.x = torch.from_numpy(x_np).float()
        if not use_rotation:
            self.x = Entity.project(self.x, Entity.NO_ROT_POINT_LAYOUT)
        self.type_ids = torch.tensor(
            [node.type_id for node in self.items], dtype=torch.long  # type: ignore
        )
        self.role_ids = torch.tensor(
            [node.role.value for node in self.items], dtype=torch.long
        )
        self.entity_ids = torch.tensor(
            [self._entity_index[node.entity] for node in self.items], dtype=torch.long
        )

    @property
    def _entity_index(self) -> dict[str, int]:
        """Entity name -> row-group id, in order of first appearance."""
        index: dict[str, int] = {}
        for node in self.items:
            index.setdefault(node.entity, len(index))
        return index

    def update_nodes(self, start: DCScene, goal: DCScene):
        for node in self.items:
            if node.mode == ENMode.START:
                node.data = start.get(node.entity)
            elif node.mode == ENMode.GOAL:
                node.data = goal.get(node.entity)
            elif node.mode == ENMode.SAMPLE:
                assert node.con is not None
                node.data = node.con.sample(node.entity)
