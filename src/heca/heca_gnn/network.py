from dataclasses import dataclass
from typing import Sequence
import torch
from torch import nn
from torch.distributions import Categorical
from torch_geometric.data import HeteroData
from torch_geometric.nn import GINEConv, GINConv

from heca.misc import hardware
from heca.misc.base import Configurable


class StepMixBlock(nn.Module):
    """GIN layer for entity→stepmix→entity edges (8-dim edge features)."""

    def __init__(self, dim: int):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.conv = GINEConv(nn=self.nn, edge_dim=8)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        return self.conv(x, edge_index, edge_attr) + x


class TapasBlock(nn.Module):
    """GIN layer for entity→tapas→entity edges (no edge features)."""

    def __init__(self, dim: int):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.conv = GINConv(nn=self.nn)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.conv(x, edge_index) + x


class SummaryBlock(nn.Module):
    """GINE layer for entity→summary→option edges (7-dim edge features)."""

    def __init__(self, dim: int):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.conv = GINEConv(nn=self.nn, edge_dim=7)

    def forward(
        self,
        x_entity: torch.Tensor,
        x_option: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        return self.conv((x_entity, x_option), edge_index, edge_attr)


class OptionReadout(nn.Module):
    """Separate MLPs for Actor and Critic to avoid task interference."""

    def __init__(self, dim: int):
        super().__init__()
        self.actor_net = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1)
        )

        self.critic_net = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        actor_out = self.actor_net(x)  # [num_options, 1]
        logits = actor_out.view(1, -1)  # [1, num_options]

        pooled_x = x.mean(dim=0, keepdim=True)  # [1, feature_dim]
        value = self.critic_net(pooled_x).squeeze(-1)  # scalar [1]

        return logits, value


class Network(Configurable, nn.Module):
    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        input_dim: int = 64  # 13 + max_state_count
        feature_dim: int = 128
        type_embed_dim: int = 8
        max_entity_types: int = 64  # just an upper bound
        num_stepmix_layers: int = 1
        num_tapas_layers: int = 1

    def __init__(self, cfg: Config):
        nn.Module.__init__(self)
        self.cfg = cfg

        self.type_embedding = nn.Embedding(cfg.max_entity_types, cfg.type_embed_dim)

        total_input_dim = cfg.input_dim + cfg.type_embed_dim

        self.entity_encoder = nn.Sequential(
            nn.Linear(total_input_dim, cfg.feature_dim),
            nn.LayerNorm(cfg.feature_dim),
            nn.ReLU(),
        )

        self.stepmix_layers = nn.ModuleList(
            [StepMixBlock(cfg.feature_dim) for _ in range(cfg.num_stepmix_layers)]
        )

        self.tapas_layers = nn.ModuleList(
            [TapasBlock(cfg.feature_dim) for _ in range(cfg.num_tapas_layers)]
        )

        self.summary_layer = SummaryBlock(cfg.feature_dim)
        self.option_readout = OptionReadout(cfg.feature_dim)

    def forward(self, data: HeteroData) -> tuple[torch.Tensor, torch.Tensor]:
        type_embeds = self.type_embedding(
            data["entity"].type_ids
        )  # [N, type_embed_dim]
        entity_x = torch.cat([data["entity"].x, type_embeds], dim=-1)

        entity_x = self.entity_encoder(entity_x)

        stepmix_idx = data[("entity", "stepmix", "entity")].edge_index
        stepmix_attr = data[("entity", "stepmix", "entity")].edge_attr
        for layer in self.stepmix_layers:
            entity_x = layer(entity_x, stepmix_idx, stepmix_attr)

        tapas_idx = data[("entity", "tapas", "entity")].edge_index
        for layer in self.tapas_layers:
            entity_x = layer(entity_x, tapas_idx)

        summary_idx = data[("entity", "summary", "option")].edge_index
        summary_attr = data[("entity", "summary", "option")].edge_attr
        option_x = self.summary_layer(
            entity_x, data["option"].x, summary_idx, summary_attr
        )

        logits, value = self.option_readout(option_x)
        return logits, value

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
            logits, value = self.forward(data)  # value is now [1] scalar!
            dist = Categorical(logits=logits)

            action = actions[i : i + 1]
            logprob = dist.log_prob(action)  # [1]
            entropy = dist.entropy()  # [1]

            logprobs.append(logprob)
            state_values.append(value)  # No indexing! Just append the scalar.
            entropies.append(entropy)

        return (
            torch.cat(logprobs).to(hardware.device),
            torch.cat(state_values).to(hardware.device),
            torch.cat(entropies).to(hardware.device),
        )
