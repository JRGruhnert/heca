from torch import nn

from heca.graphs.roles import ENRole
from heca.graphs.data import HecaData
from heca.graphs.edges.condition_edges import ConditionEdges
from heca.graphs.edges.translation_edges import TranslationEdges
from heca.graphs.nodes.entity_nodes import EntityNodes
from heca.heca_gnn.modules.condition import ConditionGatBlock, ConditionGinBlock
from heca.heca_gnn.modules.encoders.encoder import EncodedRows, EncoderBlock
from heca.heca_gnn.modules.hyperedge import TypedHyperedgeLayer
from heca.heca_gnn.modules.interaction import IdentityBlock
from heca.heca_gnn.modules.translation import TranslationBlock


class RootNetwork(nn.Module):
    def __init__(
        self,
        name: str,
        feature_dim: int,
        condition_gat: bool,
        hyperedge: bool,
        effects: bool,
    ):
        nn.Module.__init__(self)
        self.name = name
        self.layers = nn.ModuleDict(
            {
                "encoder": EncoderBlock(feature_dim, effects),
                "condition": (
                    ConditionGatBlock(feature_dim)
                    if condition_gat
                    else ConditionGinBlock(feature_dim)
                ),
                "hyperedge": (
                    TypedHyperedgeLayer(feature_dim, num_roles=ENRole.size())
                    if hyperedge
                    else IdentityBlock()
                ),
                "translation": TranslationBlock(feature_dim),
            }
        )

    def forward(self, data: HecaData) -> EncodedRows:
        x: EncodedRows = self.layers["encoder"](data)

        entity_x = self.layers["condition"](
            x.comp,
            x.entity,
            data[ConditionEdges.type].edge_index,
            data[ConditionEdges.type].edge_attr,
        )

        entity_x = self.layers["hyperedge"](
            entity_x,
            data[EntityNodes.type].role_ids,
        )

        entity_x = self.layers["translation"](
            entity_x,
            data[TranslationEdges.type].edge_index,
        )

        return x._replace(entity=entity_x)
