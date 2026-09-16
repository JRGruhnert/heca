import numpy as np
import torch

from heca.graphs.nodes.entity_nodes import EntityNodes
from heca.graphs.nodes.node import OptionNode
from heca.graphs.nodes.node_set import NodeSet


def _zscore(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Standardise across options; an all-equal input maps to all zeros."""
    std = float(values.std())
    if std < eps:
        return np.zeros_like(values)
    return (values - values.mean()) / std


class OptionNodes(NodeSet[OptionNode]):
    type = "option"

    FEATURE_DIM = 4
    RECENCY_HALF_LIFE: float = 4.0  # steps until a use decays to half weight

    def decay(self):
        """One elapsed step: let every option's recency fade (call every step)."""
        factor = 0.5 ** (1.0 / self.RECENCY_HALF_LIFE)
        for node in self.items:
            node.recency *= factor

    def reset_stats(self):
        """Begin a new episode: the statistics are episode-local, not historic."""
        for node in self.items:
            node.exec_count = 0
            node.recency = 0.0

    def record(self, idx: int):
        """Count one execution of the option at ``idx``."""
        node = self.idx_get(idx)
        node.exec_count += 1
        node.recency = 1.0

    def build(self, ns_entity: EntityNodes, use_rotation: bool = True):
        self.gated = self.gates(ns_entity)
        self.x = self.statistics()

    def statistics(self) -> torch.Tensor:
        """(n_options, FEATURE_DIM): z-scored count / group / recency + gate."""
        per_option = np.array(
            [node.exec_count for node in self.items], dtype=np.float32
        )
        recency = np.array([node.recency for node in self.items], dtype=np.float32)
        gated = self.gated.numpy().astype(np.float32)

        per_group: dict[str, int] = {}
        for node in self.items:
            per_group[node.model.tag] = (
                per_group.get(node.model.tag, 0) + node.exec_count
            )
        group_of_option = np.array(
            [per_group[node.model.tag] for node in self.items], dtype=np.float32
        )

        return torch.from_numpy(
            np.stack(
                [
                    _zscore(per_option),
                    _zscore(group_of_option),
                    _zscore(recency),
                    gated,
                ],
                axis=1,
            )
        ).float()

    def gates(self, ns_entity: EntityNodes) -> torch.Tensor:
        gated = [not self._gated(o, ns_entity) for o in self.items]
        return torch.tensor(gated, dtype=torch.float32)

    def _gated(self, o: OptionNode, ns_entity: EntityNodes) -> bool:
        for src in o.sources[EntityNodes.type]:
            post = ns_entity.get_by_key(src)
            assert post.con is not None
            if not post.con.test(post.entity, post.data):
                return False
            for src2 in post.sources[EntityNodes.type]:
                pre = ns_entity.get_by_key(src2)
                assert pre.con is not None
                if not pre.con.test(pre.entity, pre.data):
                    return False
        return True
