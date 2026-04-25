import torch
from torch import nn

from torch_geometric.nn import GINConv

from hoopgn.misc.classes import ConfigurableClass
from hoopgn.misc.mlp import UnactivatedMLP


class ActorReadoutNetwork(ConfigurableClass, nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__(feature_dim)
        self.actor_readout = GINConv(
            nn=UnactivatedMLP(feature_dim, feature_dim),
        )

    def readout(
        self, x: torch.Tensor, x_dict: dict, edge_index_dict: dict
    ) -> torch.Tensor:
        logits: torch.Tensor = self.readout_net(
            x=(x, x_dict["actor"]),
            edge_index=edge_index_dict[("task", "task-actor", "actor")],
        )
        return logits.view(1, -1)
