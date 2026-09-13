import numpy as np
import torch

from heca.data.data import DCScene
from heca.data.entity import Entity
from heca.graphs.nodes.node import CanonicalNode
from heca.graphs.nodes.node_set import NodeSet
from heca.graphs.roles import ROLE_CURRENT, ROLE_GOAL


class CanonicalNodes(NodeSet[CanonicalNode]):
    type = "canonical"

    def __init__(self, entities: dict[str, Entity]):
        super().__init__()
        self.entities = entities
        for label, entity in self.entities.items():
            for role in (ROLE_CURRENT, ROLE_GOAL):
                self.add(
                    f"{label}{role}",
                    CanonicalNode(
                        entity=label,
                        type_id=entity.cfg.type_id,
                        n_states=entity.cfg.n_states,
                        role=role,
                    ),
                )

    def build(self, start: DCScene, goal: DCScene):
        for node in self.items:
            if node.role == ROLE_CURRENT:
                node.data = start.get(node.entity)
            elif node.role == ROLE_GOAL:
                node.data = goal.get(node.entity)
            else:
                continue

        x_np = np.stack([node.data.feature for node in self.items], axis=0)
        self.x = torch.from_numpy(x_np).float()
        self.type_ids = torch.tensor(
            [node.type_id for node in self.items], dtype=torch.long  # type: ignore
        )
