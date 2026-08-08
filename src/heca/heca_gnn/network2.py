from dataclasses import dataclass
import torch
from torch import nn
from torch_geometric.data import HeteroData

from heca.heca_gnn.network import Network, StepMixBlock, TapasBlock, SummaryBlock


class OptionMemory(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.gru_cell = nn.GRUCell(feature_dim, feature_dim)

        self._memory_initialized = False

    def _init_memory(self, num_options: int, device: torch.device):
        buf = torch.zeros(num_options, self.feature_dim, device=device)
        self.register_buffer("memory", buf, persistent=False)
        self._memory_initialized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._memory_initialized:
            self._init_memory(x.size(0), x.device)
        hidden = self.gru_cell(x, self.memory)
        self.memory.copy_(hidden.detach())
        return x + hidden

    def reset_memory(self) -> None:
        self.memory.zero_()


class OptionInteraction(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = x.unsqueeze(0)
        attended, _ = self.attn(xs, xs, xs)
        return self.norm(xs.squeeze(0) + attended.squeeze(0))


class OptionReadout(nn.Module):
    def __init__(self, dim: int, hidden_ratio: float = 0.5):
        super().__init__()
        hidden_dim = max(int(dim * hidden_ratio), 16)

        self.shared = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
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
    @dataclass(kw_only=True)
    class Config(Network.Config):
        mobility_feat_dim: int = 3
        feature_dim: int = 256
        encoder_depth: int = 3
        gnn_mlp_depth: int = 3
        use_option_interaction: bool = True
        use_option_memory: bool = True
        attn_heads: int = 4
        readout_hidden_ratio: float = 0.5

    def __init__(self, cfg: Config):
        nn.Module.__init__(self)
        self.cfg = cfg

        total_input_dim = cfg.input_feat_dim + cfg.mobility_feat_dim

        # TODO: Entity Encoder
        self.entity_encoder = nn.Sequential(
            nn.LayerNorm(total_input_dim),
            nn.Linear(total_input_dim, cfg.feature_dim),
            nn.LayerNorm(cfg.feature_dim),
            nn.ReLU(),
        )
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
            self.interaction_layer = OptionInteraction(cfg.feature_dim, cfg.attn_heads)
        else:
            self.interaction_layer = None

        if cfg.use_option_memory:
            self.memory_layer = OptionMemory(cfg.feature_dim)
        else:
            self.memory_layer = None

        self.option_readout = OptionReadout(cfg.feature_dim, cfg.readout_hidden_ratio)

    def forward(self, data: HeteroData) -> tuple[torch.Tensor, torch.Tensor]:

        entity_x = torch.cat([data["entity"].x, data["entity"].type_ids], dim=-1)

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

        if self.interaction_layer is not None:
            option_x = self.interaction_layer(option_x)

        if self.memory_layer is not None:
            option_x = self.memory_layer(option_x)

        logits, value = self.option_readout(option_x)

        return logits, value

    def reset_memory(self):
        if self.memory_layer is not None:
            self.memory_layer.reset_memory()
