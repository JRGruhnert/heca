from heca.graphs.edges.edge_set import EdgeSet
from heca.graphs.nodes.entity_nodes import EntityNodes
from heca.graphs.nodes.node import EntityNode, OptionNode
from heca.graphs.nodes.option_nodes import OptionNodes


class SummaryEdges(EdgeSet[EntityNode, OptionNode]):
    has_attrs: bool = False
    type = (EntityNodes.type, "summary", OptionNodes.type)
