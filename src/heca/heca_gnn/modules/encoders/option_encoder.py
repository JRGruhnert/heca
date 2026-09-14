import torch
from torch import nn


class OptionEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, use_effects: bool = True):
        super().__init__()
        self.use_effects = use_effects
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim // 2),
            nn.GELU(),
            nn.Linear(out_dim // 2, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_effects:
            x = torch.zeros_like(x)
        return self.net(x)
