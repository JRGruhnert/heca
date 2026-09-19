import numpy as np
import torch

from heca.graphs.edges.edge_set import DEFAULT_TERMS, EdgeSet
from heca.graphs.nodes.comp_nodes import CompNodes
from heca.graphs.nodes.entity_nodes import EntityNodes
from heca.graphs.nodes.node import CompNode, EntityNode
from heca.graphs.nodes.node_set import NodeSet
from heca.graphs.roles import ENRole


class ConditionEdges(EdgeSet[CompNode, EntityNode]):
    type = (CompNodes.type, "condition", EntityNodes.type)

    @classmethod
    def edge_dim(
        cls,
        use_rotation: bool = True,
        terms: tuple[str, ...] = DEFAULT_TERMS,
        goal_residual: bool = False,
    ) -> int:
        """Residual blocks (one per value tested) plus the log mixture weight."""
        blocks = 2 if goal_residual else 1
        return blocks * cls.residual_dim(terms, use_rotation) + 1

    @staticmethod
    def goal_features(
        dnset: NodeSet[EntityNode], dst_list: tuple[int, ...]
    ) -> np.ndarray:
        goal_rows = {
            node.entity: i
            for i, node in enumerate(dnset.items)
            if node.role == ENRole.GOAL
        }
        missing = sorted(
            {
                node.entity
                for node in (dnset.idx_get(int(j)) for j in dst_list)
                if node.entity not in goal_rows
            }
        )
        if missing:
            raise ValueError(
                "the goal residual needs the goal row of every destination "
                f"entity; missing: {', '.join(missing)}"
            )
        return EdgeSet.gather_features(
            dnset,
            [goal_rows[dnset.idx_get(int(j)).entity] for j in dst_list],
        )

    def build(
        self,
        snset: NodeSet[CompNode],
        dnset: NodeSet[EntityNode],
        use_rotation: bool = True,
        terms: tuple[str, ...] = DEFAULT_TERMS,
        goal_residual: bool = False,
    ):
        self.reset()
        self.edges_from_sets(snset, dnset)
        src_list, dst_list = zip(*self.edges)
        self.edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

        x_src = self.gather_features(snset, src_list)
        x_dst = self.gather_features(dnset, dst_list)
        assert all(
            snset.items[i].n_states == dnset.items[j].n_states for i, j in self.edges
        ), "condition edges must share a state vocabulary"
        weights = np.fromiter(
            (snset.items[int(i)].weight for i in src_list),
            dtype=np.float32,
            count=len(src_list),
        )
        log_weights = np.log(weights + 1e-15)

        radius = None
        if "rbf" in terms:
            radius = np.empty(len(src_list), dtype=np.float64)
            for i, j in enumerate(dst_list):
                node = dnset.items[int(j)]
                if node.con is None:
                    raise ValueError(
                        "the 'rbf' term needs the destination entity's condition "
                        f"to know its acceptance radius ({node.entity} has none)"
                    )
                radius[i] = node.con.entities[node.entity].acceptance_radius(
                    use_rotation
                )

        blocks = [
            self.residual_batch(
                x_src,
                x_dst,
                use_rotation=use_rotation,
                terms=terms,
                radius=radius,
            )
        ]
        if goal_residual:
            # the goal value under the same component, on every condition edge.
            # On a row whose own value is the current value (precondition rows,
            # start-anchored postcondition rows) the two halves differ by exactly
            # (current - goal) / sigma of that component, so the pair carries the
            # goal-versus-current distance without a second relation.
            blocks.append(
                self.residual_batch(
                    x_src,
                    self.goal_features(dnset, dst_list),
                    use_rotation=use_rotation,
                    terms=terms,
                    radius=radius,
                )
            )

        attr = np.concatenate([*blocks, log_weights[:, None]], axis=-1)
        self.attrs = list(attr)
        self.edge_attr = torch.from_numpy(attr).float()
