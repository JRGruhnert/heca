from dataclasses import dataclass
from typing import Sequence
import torch
from torch import nn
from torch.distributions import Categorical
from torch_geometric.data import HeteroData
from torch_geometric.nn import GINEConv, GINConv

from heca.misc import hardware
from heca.misc.base import Configurable


def _make_gnn_mlp(dim: int, num_layers: int) -> nn.Sequential:
    layers = []
    for _ in range(num_layers):
        layers.append(nn.Linear(dim, dim))
        layers.append(nn.LayerNorm(dim))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(dim, dim))
    return nn.Sequential(*layers)


class StepMixBlock(nn.Module):
    """GINE layer for entity→stepmix→entity edges (8-dim edge features)."""

    def __init__(self, dim: int, num_layers: int = 2):
        super().__init__()
        self.nn = _make_gnn_mlp(dim, num_layers)
        self.conv = GINEConv(nn=self.nn, edge_dim=8)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        return self.conv(x, edge_index, edge_attr) + x


class TapasBlock(nn.Module):
    """GIN layer for entity→tapas→entity edges (no edge features)."""

    def __init__(self, dim: int, num_layers: int = 2):
        super().__init__()
        self.nn = _make_gnn_mlp(dim, num_layers)
        self.conv = GINConv(nn=self.nn)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.conv(x, edge_index) + x


class SummaryBlock(nn.Module):
    """GINE layer for entity→summary→option edges (7-dim edge features)."""

    def __init__(self, dim: int, num_layers: int = 2):
        super().__init__()
        self.nn = _make_gnn_mlp(dim, num_layers)
        self.conv = GINEConv(nn=self.nn, edge_dim=7)

    def forward(
        self,
        x_entity: torch.Tensor,
        x_option: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        return self.conv((x_entity, x_option), edge_index, edge_attr)


class Network(Configurable, nn.Module):
    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        input_feat_dim: int = 56
        feature_dim: int = 128
        num_stepmix_layers: int = 1
        num_tapas_layers: int = 1

    def __init__(self, cfg: Config):
        nn.Module.__init__(self)
        self.cfg = cfg

    def forward(self, data: HeteroData) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def actor(self, data: HeteroData) -> torch.Tensor:
        logits, _ = self.forward(data)
        return logits

    def critic(self, data: HeteroData) -> torch.Tensor:
        _, value = self.forward(data)
        return value

    def upgrade(self, checkpoint):
        self.load_state_dict(checkpoint, strict=False)

    def evaluate(
        self, data_list: Sequence[HeteroData], actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logprobs = []
        state_values = []
        entropies = []

        for i, data in enumerate(data_list):
            logits, value = self.forward(data)
            dist = Categorical(logits=logits)

            action = actions[i : i + 1]
            logprob = dist.log_prob(action)
            entropy = dist.entropy()

            logprobs.append(logprob)
            state_values.append(value)
            entropies.append(entropy)

        return (
            torch.cat(logprobs).to(hardware.device),
            torch.cat(state_values).to(hardware.device),
            torch.cat(entropies).to(hardware.device),
        )

    def reset_memory(self):
        pass
