import numpy as np
import torch

from heca.data.entity import Entity
from heca.graphs.nodes.entity_nodes import EntityNodes
from heca.graphs.nodes.node import OptionNode
from heca.graphs.nodes.node_set import NodeSet


class OptionNodes(NodeSet[OptionNode]):
    type = "option"
    FEATURE_DIM = Entity.FEATURE_DIM

    def build(self, ns_entity: EntityNodes):
        self.x = self.effects()
        self.gated = self.gates(ns_entity)

    def effects(self) -> torch.Tensor:
        effects = [node.effect for node in self.items]
        widths = [e.shape[-1] for e in effects if e is not None]
        width = widths[0] if widths else 0
        x_np = np.zeros((len(self.items), width), dtype=np.float32)
        for i, effect in enumerate(effects):
            if effect is not None and effect.shape[-1] == width:
                x_np[i] = effect
        return torch.from_numpy(x_np).float()

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
