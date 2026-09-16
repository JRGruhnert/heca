from typing import NamedTuple

import torch
from torch import nn

from heca.graphs.data import HecaData
from heca.graphs.edges.scene_edges import SceneEdges
from heca.graphs.edges.summary_edges import SummaryEdges
from heca.graphs.nodes.option_nodes import OptionNodes

from heca.heca_gnn.modules.boundary import BoundaryNorm, NoUpdateBlock, PairNormBlock
from heca.heca_gnn.modules.encoders.encoder import EncodedRows
from heca.heca_gnn.modules.interaction import TransformerBlock
from heca.heca_gnn.modules.scene import SceneGATBlock, SceneSageBlock
from heca.heca_gnn.modules.summary import SummarySageBlock, SummaryGinBlock
from heca.heca_gnn.modules.timeline import MemoryBlock, NoMemoryBlock


class TruncOutput(NamedTuple):
    option: torch.Tensor
    state: torch.Tensor
    memory: torch.Tensor | None


class TruncNetwork(nn.Module):
    def __init__(
        self,
        name: str,
        feature_dim: int,
        option_transformer: bool,
        summary_sage: bool,
        pair_norm: bool,
        memory: bool,
    ):
        nn.Module.__init__(self)
        self.name = name
        self.feature_dim = feature_dim
        self.layers = nn.ModuleDict(
            {
                "norm": BoundaryNorm(
                    feature_dim,
                    (
                        "summary_entity",
                        "summary_option",
                        "interaction_option",
                        "scene_option",
                        "scene_state",
                        "timeline_state",
                    ),
                ),
                "pair": PairNormBlock(("option",), pair_norm),
                "summary": (
                    SummarySageBlock(feature_dim)
                    if summary_sage
                    else SummaryGinBlock(feature_dim)
                ),
                "interaction": (
                    TransformerBlock(feature_dim)
                    if option_transformer
                    else NoUpdateBlock()
                ),
                "scene": (
                    SceneSageBlock(feature_dim)
                    if option_transformer
                    else SceneGATBlock(feature_dim)
                ),
                "timeline": (MemoryBlock(feature_dim) if memory else NoMemoryBlock()),
            }
        )

    def forward(self, data: HecaData, x: EncodedRows) -> TruncOutput:
        layer = self.layers
        norm = layer["norm"]
        pair = layer["pair"]

        entity_x = norm("summary_entity", x.entity)
        option_x = norm("summary_option", x.option)
        option_x = option_x + layer["summary"](
            entity_x,
            option_x,
            data[SummaryEdges.type].edge_index,
        )
        option_x = pair("option", option_x)

        option_x = norm("interaction_option", option_x)
        option_x = option_x + layer["interaction"](
            option_x,
            data[OptionNodes.type].gated,
        )
        option_x = pair("option", option_x)

        state_x = norm("scene_state", x.state)
        state_x = state_x + layer["scene"](
            norm("scene_option", option_x),
            state_x,
            data[SceneEdges.type].edge_index,
        )

        memory_x = layer["timeline"](
            norm("timeline_state", state_x),
            data.memory.get(self.name),
        )

        return TruncOutput(option=option_x, state=state_x, memory=memory_x)
