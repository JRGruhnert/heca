from typing import Mapping, NamedTuple, Sequence

import torch
from torch import nn

from heca.data.entity import Entity
from heca.graphs.data import HecaData
from heca.heca_gnn.modules.encoders.common import SetEncoder
from heca.heca_gnn.modules.encoders.option_encoder import OptionEncoder


class EncodedRows(NamedTuple):
    entity: torch.Tensor
    comp: torch.Tensor
    option: torch.Tensor
    # optional: only present when "canonical" is in HecaEncoder.ROW_FIELDS
    canonical: torch.Tensor | None = None


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
    ROW_FIELDS = ("entity", "comp")

    def __init__(self, dim: int, use_option_effects: bool = True):
        nn.Module.__init__(self)
        self.dim = dim
        self.rows = RowEncoderGroup(dim, self.ROW_FIELDS)
        self.option_encoder = OptionEncoder(Entity.FEATURE_DIM, dim, use_option_effects)

    def forward(self, data: HecaData) -> EncodedRows:
        encoded = self.rows(
            {name: (data[name].x, data[name].type_ids) for name in self.rows.fields}
        )
        encoded["option"] = self.option_encoder(data.option.x)
        return EncodedRows(**encoded)
