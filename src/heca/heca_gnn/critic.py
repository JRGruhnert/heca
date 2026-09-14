import torch
from torch import nn

from heca.heca_gnn.modules.film import FiLM, Identity
from heca.heca_gnn.trunc import TruncOutput


class CriticNetwork(nn.Module):
    def __init__(self, feature_dim: int, use_memory: bool):
        super().__init__()

        if use_memory:
            self.film = FiLM(feature_dim)
        else:
            self.film = Identity()

        self.mlp = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, 1),
        )

    def forward(self, data: TruncOutput) -> torch.Tensor:
        state_x = self.film(data.state, data.memory)
        return self.mlp(state_x).reshape(1)  # (1,)
