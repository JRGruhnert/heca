"""Original GNN architecture with learned type embeddings and separate
actor / critic MLPs.
"""

from dataclasses import dataclass
import torch
from torch import nn
from torch_geometric.data import HeteroData

from heca.heca_gnn.network import (
    Network,
    StepMixBlock,
    TapasBlock,
    SummaryBlock,
)


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
        actor_out = self.actor_net(x)
        logits = actor_out.view(1, -1)

        pooled_x = x.mean(dim=0, keepdim=True)
        value = self.critic_net(pooled_x).squeeze(-1)

        return logits, value


class Network1(Network):
    """Original GNN: type embeddings → encoder → stepmix → tapas → summary → readout."""

    @dataclass(kw_only=True)
    class Config(Network.Config):
        type_embed_dim: int = 8
        feature_dim: int = 128
        max_entity_types: int = 64

    def __init__(self, cfg: Config):
        nn.Module.__init__(self)
        self.cfg = cfg

        self.type_embedding = nn.Embedding(cfg.max_entity_types, cfg.type_embed_dim)

        total_input_dim = cfg.input_feat_dim + cfg.type_embed_dim

        self.entity_encoder = nn.Sequential(
            nn.LayerNorm(total_input_dim),
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
        type_embeds = self.type_embedding(data["entity"].type_ids)
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
