from typing import Mapping, NamedTuple

import torch
from torch import nn

from heca.graphs.data import HecaData
from heca.graphs.nodes.comp_nodes import CompNodes
from heca.graphs.nodes.entity_nodes import EntityNodes
from heca.graphs.nodes.state_nodes import StateNodes
from heca.heca_gnn.modules.encoders.common import SetEncoder


class EncodedRows(NamedTuple):
    entity: torch.Tensor
    comp: torch.Tensor
    state: torch.Tensor


class RowEncoderGroup(nn.Module):
    def __init__(
        self,
        dim: int,
        fields: tuple,
        rotation: bool = True,
        logstd: dict[str, bool] | None = None,
    ):
        super().__init__()
        self.fields = tuple(fields)
        logstd = logstd if logstd is not None else {f: True for f in self.fields}
        self.encoders = nn.ModuleDict(
            {f: SetEncoder(dim, rotation, logstd[f]) for f in self.fields}
        )

    def forward(
        self, rows: Mapping[str, tuple[torch.Tensor, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        return {f: self.encoders[f].encode(*rows[f]) for f in self.fields}


class EncoderBlock(nn.Module):
    ROW_FIELDS = (EntityNodes.type, CompNodes.type)

    def __init__(
        self,
        feature_dim: int,
        rotation: bool = True,
    ):
        nn.Module.__init__(self)
        self.rows = RowEncoderGroup(
            feature_dim,
            self.ROW_FIELDS,
            rotation,
            logstd={EntityNodes.type: False, CompNodes.type: True},
        )
        self.state_encoder = nn.Linear(StateNodes.FEATURE_DIM, feature_dim)

    def forward(self, data: HecaData) -> EncodedRows:
        encoded = self.rows(
            {name: (data[name].x, data[name].type_ids) for name in self.rows.fields}
        )
        encoded[StateNodes.type] = self.state_encoder(data[StateNodes.type].x)
        return EncodedRows(**encoded)
