import torch
from torch import nn


class OptionEncoder(nn.Module):
    """Encodes the option rows (the option effects).

    ``use_effects=False`` zeroes them before the norm, so the ablation really
    removes the effect information instead of only hiding it behind a bias - the
    caller just hands over the raw option features.
    """

    def __init__(self, in_dim: int, out_dim: int, use_effects: bool = True):
        super().__init__()
        self.use_effects = use_effects
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_effects:
            x = torch.zeros_like(x)
        return self.net(x)
