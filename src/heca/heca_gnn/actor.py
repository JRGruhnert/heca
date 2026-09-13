from dataclasses import dataclass

import torch
from torch import nn

from heca.misc.base import Configurable


class ActorNetwork(Configurable, nn.Module):

    def __init__(self, dim: int, hidden_ratio: float = 0.5):
        nn.Module.__init__(self)

        hidden_dim = max(int(dim * hidden_ratio), 16)

        self.actor_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        actor_out = self.actor_head(x)
        return actor_out.view(1, -1)
