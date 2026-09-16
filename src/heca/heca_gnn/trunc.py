from typing import NamedTuple

import torch
from torch import nn

from heca.graphs.data import HecaData
from heca.graphs.edges.scene_edges import SceneEdges
from heca.graphs.edges.summary_edges import SummaryEdges
from heca.graphs.nodes.option_nodes import OptionNodes

from heca.graphs.nodes.state_nodes import StateNodes
from heca.heca_gnn.modules.encoders.encoder import EncodedRows
from heca.heca_gnn.modules.interaction import IdentityBlock, TransformerBlock
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
        memory: bool,
    ):
        nn.Module.__init__(self)
        self.name = name
        self.feature_dim = feature_dim
        self.layers = nn.ModuleDict(
            {
                "summary": (
                    SummarySageBlock(feature_dim)
                    if summary_sage
                    else SummaryGinBlock(feature_dim)
                ),
                "interaction": (
                    TransformerBlock(feature_dim)
                    if option_transformer
                    else IdentityBlock()
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
        n_option = data[OptionNodes.type].x.shape[0]
        option_x = x.entity.new_zeros(n_option, self.feature_dim)
        option_x = layer["summary"](
            x.entity,
            option_x,
            data[SummaryEdges.type].edge_index,
        )

        option_x = layer["interaction"](
            option_x,
            data[OptionNodes.type].gated,
        )

        n_state = data[StateNodes.type].x.shape[0]
        state_x = x.entity.new_zeros(n_state, self.feature_dim)
        state_x = layer["scene"](
            option_x,
            state_x,
            data[SceneEdges.type].edge_index,
            data[SceneEdges.type].edge_attr,
        )
        memory_x = layer["timeline"](state_x, data.memory.get(self.name))

        return TruncOutput(option=option_x, state=state_x, memory=memory_x)
