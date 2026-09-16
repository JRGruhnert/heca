import torch
from torch import nn


class StateEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, statistics: bool):
        super().__init__()
        self.statistics = statistics
        self.out_dim = out_dim
        self.net = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.statistics:
            return x.new_zeros(x.shape[0], self.out_dim)
        return self.net(x)
