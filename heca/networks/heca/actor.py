import torch
from torch import nn

from torch_geometric.nn import GINConv

from heca.networks.mlp import UnactivatedMLP
from torch_geometric.data import HeteroData


class ActorReadoutNetwork(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.actor_readout = GINConv(
            nn=UnactivatedMLP(feature_dim, feature_dim),
        )

    def forward(self, data: HeteroData) -> torch.Tensor:
        logits: torch.Tensor = self.actor_readout(
            x=(data["task"].x, data["actor"].x),
            edge_index=data[("task", "task-actor", "actor")].edge_index,
        )
        return logits.view(1, -1)
