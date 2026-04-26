import torch
from torch import nn

from torch_geometric.nn import GINConv

from hoopgn.networks.mlp import UnactivatedMLP
from torch_geometric.data import HeteroData


class CriticReadoutNetwork(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.readout_net = GINConv(
            nn=UnactivatedMLP(feature_dim, feature_dim),
        )

    def forward(self, data: HeteroData) -> torch.Tensor:
        value: torch.Tensor = self.readout_net(
            x=(data["task"].x, data["critic"].x),
            edge_index=data[("task", "task-critic", "critic")].edge_index,
        )
        return value.squeeze(-1)
