"""Extended GNN with explicit mobility features, deeper encoder, cross‑option
self‑attention and a shared readout head for federated learning.
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


class OptionInteraction(nn.Module):
    """Self-attention over option nodes for cross-option reasoning."""

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, option_x: torch.Tensor) -> torch.Tensor:
        if option_x.size(0) <= 1:
            return option_x
        x = option_x.unsqueeze(0)
        attended, _ = self.attn(x, x, x)
        return self.norm(x.squeeze(0) + attended.squeeze(0))


class OptionReadout(nn.Module):
    """Shared processing MLP with thin actor/critic heads for FL."""

    def __init__(self, dim: int, hidden_ratio: float = 0.5):
        super().__init__()
        hidden_dim = max(int(dim * hidden_ratio), 16)

        self.shared = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Linear(dim, hidden_dim), nn.ReLU(),
        )

        self.actor_head = nn.Linear(hidden_dim, 1)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared = self.shared(x)

        actor_out = self.actor_head(shared)
        logits = actor_out.view(1, -1)

        pooled = shared.mean(dim=0, keepdim=True)
        value = self.critic_head(pooled).squeeze(-1)

        return logits, value


class Network2(Network):
    """Extended GNN: mobility features → deeper encoder → stepmix → tapas
    → summary → cross‑option attention → shared readout.
    """

    @dataclass(kw_only=True)
    class Config(Network.Config):
        mobility_feat_dim: int = 3
        feature_dim: int = 256
        encoder_depth: int = 3
        gnn_mlp_depth: int = 3
        use_option_interaction: bool = True
        attn_heads: int = 4
        readout_hidden_ratio: float = 0.5

    def __init__(self, cfg: Config):
        nn.Module.__init__(self)
        self.cfg = cfg

        total_input_dim = cfg.input_feat_dim + cfg.mobility_feat_dim

        encoder_layers = [nn.LayerNorm(total_input_dim)]
        for i in range(cfg.encoder_depth):
            in_dim = total_input_dim if i == 0 else cfg.feature_dim
            encoder_layers.append(nn.Linear(in_dim, cfg.feature_dim))
            if i < cfg.encoder_depth - 1:
                encoder_layers.append(nn.LayerNorm(cfg.feature_dim))
                encoder_layers.append(nn.ReLU())
        self.entity_encoder = nn.Sequential(*encoder_layers)

        self.stepmix_layers = nn.ModuleList(
            [
                StepMixBlock(cfg.feature_dim, cfg.gnn_mlp_depth)
                for _ in range(cfg.num_stepmix_layers)
            ]
        )

        self.tapas_layers = nn.ModuleList(
            [
                TapasBlock(cfg.feature_dim, cfg.gnn_mlp_depth)
                for _ in range(cfg.num_tapas_layers)
            ]
        )

        self.summary_layer = SummaryBlock(cfg.feature_dim, cfg.gnn_mlp_depth)

        if cfg.use_option_interaction:
            self.option_interaction = OptionInteraction(cfg.feature_dim, cfg.attn_heads)
        else:
            self.option_interaction = None

        self.option_readout = OptionReadout(cfg.feature_dim, cfg.readout_hidden_ratio)

    def forward(self, data: HeteroData) -> tuple[torch.Tensor, torch.Tensor]:
        entity_x = torch.cat([data["entity"].x, data["entity"].mobility], dim=-1)

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

        if self.option_interaction is not None:
            option_x = self.option_interaction(option_x)

        logits, value = self.option_readout(option_x)

        return logits, value
