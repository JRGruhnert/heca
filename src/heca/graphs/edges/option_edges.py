import torch

from heca.graphs.edges.edge_set import EdgeSet
from heca.graphs.nodes.entity_nodes import EntityNodes
from heca.graphs.nodes.node import EntityNode, OptionNode
from heca.graphs.nodes.option_nodes import OptionNodes


class OptionEdges(EdgeSet[OptionNode, OptionNode]):
    has_attrs: bool = False
    type = (OptionNodes.type, "relation", OptionNodes.type)

    def build(self, snset: OptionNodes, ns_entity: EntityNodes):
        self.reset()
        labels = self.entity_labels(snset, ns_entity)
        tags = [node.model.tag for node in snset.items]
        for i in range(len(snset.items)):
            for j in range(i + 1, len(snset.items)):
                if tags[i] == tags[j] or not labels[i] & labels[j]:
                    continue
                self.add(i, j)
                self.add(j, i)

        if self.edges:
            src, dst = zip(*self.edges)
            self.edge_index = torch.tensor([src, dst], dtype=torch.long)
        else:
            self.edge_index = torch.empty((2, 0), dtype=torch.long)
        self.edge_attr = torch.empty((2, 0), dtype=torch.long)

    @staticmethod
    def entity_labels(snset: OptionNodes, ns_entity: EntityNodes) -> list[set[str]]:
        """The entity labels each option aggregates, over all of its rows."""
        out: list[set[str]] = []
        for node in snset.items:
            labels: set[str] = set()
            for key in node.sources.get(EntityNodes.type, ()):
                if not ns_entity.has_key(key):
                    continue
                src = ns_entity.get_by_key(key)
                if isinstance(src, EntityNode):
                    labels.add(src.entity)
            out.append(labels)
        return out
