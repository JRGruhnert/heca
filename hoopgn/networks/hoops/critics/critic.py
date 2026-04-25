import torch
from torch import nn


from hoopgn.misc.classes import ConfigurableClass
from hoopgn.misc.mlp import UnactivatedMLP


class CriticReadoutNetwork(ConfigurableClass, nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__(feature_dim)
        self.readout_net = GINConv(
            nn=UnactivatedMLP(feature_dim, feature_dim),
        )

    def forward(
        self, x: torch.Tensor, x_dict: dict, edge_index_dict: dict
    ) -> torch.Tensor:
        value: torch.Tensor = self.readout_net(
            x=(x, x_dict["critic"]),
            edge_index=edge_index_dict[("task", "task-critic", "critic")],
        )
        return value.squeeze(-1)
