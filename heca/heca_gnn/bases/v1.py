from dataclasses import dataclass

import torch
from torch import nn

from torch_geometric.nn import GINConv, GINEConv

from heca.heca_gnn.bases.base import BaseNetwork
from heca.heca_gnn.mlp import GinStandardMLP
from torch_geometric.data import HeteroData


class V1Network(BaseNetwork):
    @dataclass(kw_only=True)
    class Config(BaseNetwork.Config):
        pass

    def __init__(self, cfg: BaseNetwork.Config):
        super().__init__(cfg)
        self.cfg = cfg

        self.state_state_gin = GINConv(
            nn=GinStandardMLP(
                in_dim=self.cfg.feature_dim,
                out_dim=self.cfg.feature_dim,
                hidden_dim=self.cfg.feature_dim,
            ),
        )

        self.state_skill_gin = GINEConv(
            nn=GinStandardMLP(
                in_dim=self.cfg.feature_dim,
                out_dim=self.cfg.feature_dim,
                hidden_dim=self.cfg.feature_dim,
            ),
            edge_dim=2,
        )

    def forward(self, data: HeteroData) -> torch.Tensor:
        x1 = self.state_state_gin(
            x=(data["goal"].x, data["obs"].x),
            edge_index=data[("goal", "goal-obs", "obs")].edge_index,
            # edge_attr=data[("goal", "goal-obs", "obs")].edge_attr,
        )
        x2 = self.state_skill_gin(
            x=(x1, data["task"].x),
            edge_index=data[("obs", "obs-task", "task")].edge_index,
            edge_attr=data[("obs", "obs-task", "task")].edge_attr,
        )

        return x2
