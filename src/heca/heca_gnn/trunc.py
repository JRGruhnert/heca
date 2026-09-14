from typing import NamedTuple

import torch
from torch import nn

from heca.graphs.data import HecaData
from heca.graphs.edges.scene_edges import SceneEdges
from heca.graphs.edges.summary_edges import SummaryEdges
from heca.graphs.nodes.option_nodes import OptionNodes

from heca.heca_gnn.modules.encoders.encoder import EncodedRows
from heca.heca_gnn.modules.interaction import IdentityBlock, TransformerBlock
from heca.heca_gnn.modules.overview import SceneGNNBlock
from heca.heca_gnn.modules.summary import SummaryGcnBlock, SummaryGinBlock
from heca.heca_gnn.modules.timeline import MemoryBlock, NoMemoryBlock


class TruncOutput(NamedTuple):
    option: torch.Tensor
    state: torch.Tensor
    memory: torch.Tensor | None


class TruncNetwork(nn.Module):
    """The option rows and the scene row, from the entity rows of the root."""

    def __init__(
        self,
        name: str,
        feature_dim: int,
        option_transformer: bool,
        summary_gcn: bool,
        memory: bool,
    ):
        nn.Module.__init__(self)
        self.name = name
        self.layers = nn.ModuleDict(
            {
                "summary": (
                    SummaryGcnBlock(feature_dim)
                    if summary_gcn
                    else SummaryGinBlock(feature_dim)
                ),
                "interaction": (
                    TransformerBlock(feature_dim)
                    if option_transformer
                    else IdentityBlock()
                ),
                "scene": SceneGNNBlock(feature_dim),
                "timeline": (MemoryBlock(feature_dim) if memory else NoMemoryBlock()),
            }
        )

    def forward(self, data: HecaData, x: EncodedRows) -> TruncOutput:
        layer = self.layers
        option_x = layer["summary"](
            x.entity,
            x.option,
            data[SummaryEdges.type].edge_index,
        )

        option_x = layer["interaction"](
            option_x,
            data[OptionNodes.type].gated,
        )

        state_x = layer["scene"](
            option_x,
            x.state,
            data[SceneEdges.type].edge_index,
            data[SceneEdges.type].edge_attr,
        )
        memory_x = layer["timeline"](state_x, data.memory.get(self.name))

        return TruncOutput(option=option_x, state=state_x, memory=memory_x)
