from torch import nn

from heca.graphs.roles import ENRole
from heca.graphs.data import HecaData
from heca.graphs.edges.condition_edges import ConditionEdges
from heca.graphs.edges.translation_edges import TranslationEdges
from heca.graphs.nodes.entity_nodes import EntityNodes
from heca.heca_gnn.modules.boundary import BoundaryNorm, NoUpdateBlock, PairNormBlock
from heca.heca_gnn.modules.condition import ConditionGatBlock, ConditionGinBlock
from heca.heca_gnn.modules.encoders.encoder import EncodedRows, EncoderBlock
from heca.heca_gnn.modules.hyperedge import TypedHyperedgeLayer
from heca.heca_gnn.modules.translation import TranslationBlock


class RootNetwork(nn.Module):
    def __init__(
        self,
        name: str,
        feature_dim: int,
        condition_gat: bool,
        hyperedge: bool,
        statistics: bool,
        pair_norm: bool,
        rotation: bool,
    ):
        nn.Module.__init__(self)
        self.name = name
        self.layers = nn.ModuleDict(
            {
                "encoder": EncoderBlock(feature_dim, statistics, rotation),
                "norm": BoundaryNorm(
                    feature_dim,
                    (
                        "condition_comp",
                        "condition_entity",
                        "hyperedge_entity",
                        "translation_entity",
                    ),
                ),
                "pair": PairNormBlock(("entity",), pair_norm),
                "condition": (
                    ConditionGatBlock(feature_dim)
                    if condition_gat
                    else ConditionGinBlock(feature_dim)
                ),
                "hyperedge": (
                    TypedHyperedgeLayer(feature_dim, num_roles=ENRole.size())
                    if hyperedge
                    else NoUpdateBlock()
                ),
                "translation": TranslationBlock(feature_dim),
            }
        )

    def forward(self, data: HecaData) -> EncodedRows:
        layer = self.layers
        norm = layer["norm"]
        pair = layer["pair"]
        x: EncodedRows = layer["encoder"](data)

        entity_x = norm("condition_entity", x.entity)
        entity_x = entity_x + layer["condition"](
            norm("condition_comp", x.comp),
            entity_x,
            data[ConditionEdges.type].edge_index,
            data[ConditionEdges.type].edge_attr,
        )
        entity_x = pair("entity", entity_x)

        entity_x = norm("hyperedge_entity", entity_x)
        entity_x = entity_x + layer["hyperedge"](
            entity_x,
            data[EntityNodes.type].role_ids,
            data[EntityNodes.type].type_ids,
        )
        entity_x = pair("entity", entity_x)

        entity_x = norm("translation_entity", entity_x)
        entity_x = entity_x + layer["translation"](
            entity_x,
            data[TranslationEdges.type].edge_index,
        )
        entity_x = pair("entity", entity_x)

        return x._replace(entity=entity_x)
