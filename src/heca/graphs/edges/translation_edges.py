import numpy as np

from heca.graphs.edges.edge_set import EdgeSet
from heca.graphs.nodes.entity_nodes import EntityNodes
from heca.graphs.nodes.node import EntityNode


class TranslationEdges(EdgeSet[EntityNode, EntityNode]):
    has_attrs: bool = False
    type = (EntityNodes.type, "translation", EntityNodes.type)
