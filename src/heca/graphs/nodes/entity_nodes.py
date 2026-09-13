import numpy as np
import torch

from heca.data.data import DCScene
from heca.graphs.nodes.node import EntityNode, ValueMode
from heca.graphs.nodes.node_set import NodeSet
from heca.graphs.roles import ROLE_OTHER, ROLE_POST, ROLE_PRE


class EntityNodes(NodeSet[EntityNode]):
    type = "entity"

    def build(self, start: DCScene, goal: DCScene):
        self.update_nodes(start, goal)
        x_np = np.stack([node.data.feature for node in self.items], axis=0)
        self.type_ids = torch.tensor(
            [node.type_id for node in self.items], dtype=torch.long  # type: ignore
        )
        # self.role_id = torch.tensor(
        #     [self._row_role(node) for node in self.items], dtype=torch.long
        # )
        self.x = torch.from_numpy(x_np).float()

    # def _row_role(self, node: EntityNode) -> int:
    #     if isinstance(node, SubgoalNode):
    #         return ROLE_POST  # chain shared value == the target it drives to
    #     if isinstance(node, ValueNode):
    #         return ROLE_PRE if node.vmode == ValueMode.START else ROLE_POST
    #     return ROLE_OTHER  # any other kind of row that lands in this set

    def update_nodes(self, start: DCScene, goal: DCScene):
        for node in self.items:
            if node.vmode == ValueMode.START:
                node.data = start.get(node.entity)
            elif node.vmode == ValueMode.GOAL:
                node.data = goal.get(node.entity)
            elif node.vmode == ValueMode.SAMPLE:
                node.data = node.con.sample(node.entity)
            else:
                continue
                if node.con.test(node.entity, goal):
                    node.data = goal.get(node.entity)
                else:
                    node.data = node.con.sample(node.entity)
