from torch import nn
import torch


class StateAggregation(nn.Module):
    def __init__(self, dim: int, hidden_ratio: float = 0.5):
        super().__init__()
        hidden = max(int(dim * hidden_ratio), 16)
        self.pre = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(2 * dim),
            nn.Linear(2 * dim, dim),
            nn.ReLU(),
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        """Pool one set of rows into a single vector (mean, then max).

        A set function, so it needs no edges: the caller says which rows belong
        to the slot.
        """
        rows = self.pre(rows)
        return self.head(torch.cat([rows.mean(0), rows.amax(0)], dim=-1))
