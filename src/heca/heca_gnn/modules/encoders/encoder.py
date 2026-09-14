from typing import Mapping, NamedTuple

import torch
from torch import nn

from heca.data.entity import Entity
from heca.graphs.data import HecaData
from heca.graphs.nodes.comp_nodes import CompNodes
from heca.graphs.nodes.entity_nodes import EntityNodes
from heca.graphs.nodes.option_nodes import OptionNodes
from heca.graphs.nodes.state_nodes import StateNodes
from heca.heca_gnn.modules.encoders.common import SetEncoder
from heca.heca_gnn.modules.encoders.option_encoder import OptionEncoder


class EncodedRows(NamedTuple):
    entity: torch.Tensor
    comp: torch.Tensor
    option: torch.Tensor
    state: torch.Tensor


class RowEncoderGroup(nn.Module):
    def __init__(self, dim: int, fields: tuple):
        super().__init__()
        self.fields = tuple(fields)
        self.encoders = nn.ModuleDict({f: SetEncoder(dim) for f in self.fields})

    def forward(
        self, rows: Mapping[str, tuple[torch.Tensor, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        return {f: self.encoders[f].encode(*rows[f]) for f in self.fields}


class EncoderBlock(nn.Module):
    ROW_FIELDS = (EntityNodes.type, CompNodes.type)

    def __init__(self, feature_dim: int, use_option_effects: bool = True):
        nn.Module.__init__(self)
        self.rows = RowEncoderGroup(feature_dim, self.ROW_FIELDS)
        self.option_encoder = OptionEncoder(
            OptionNodes.FEATURE_DIM, feature_dim, use_option_effects
        )
        self.state_encoder = nn.Linear(StateNodes.FEATURE_DIM, feature_dim)

    def forward(self, data: HecaData) -> EncodedRows:
        encoded = self.rows(
            {name: (data[name].x, data[name].type_ids) for name in self.rows.fields}
        )
        encoded[OptionNodes.type] = self.option_encoder(data.option.x)
        encoded[StateNodes.type] = self.state_encoder(data[StateNodes.type].x)
        return EncodedRows(**encoded)
