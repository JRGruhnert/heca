import torch
from torch import nn


class CriticNetwork(nn.Module):
    def __init__(self, dim: int, hidden_ratio: float = 0.5):
        super().__init__()
        hidden = max(int(dim * hidden_ratio), 16)

        self.project = nn.Linear(3 * dim, dim)

        self.tail = nn.Sequential(
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        canonical_x: torch.Tensor,
        cur_idx: torch.Tensor,
        goal_idx: torch.Tensor,
    ) -> torch.Tensor:
        cur = canonical_x[cur_idx]  # (E, D) — same entity order as goal
        goal = canonical_x[goal_idx]  # (E, D)
        res = cur - goal  # per-entity residual (progress toward the goal)

        stats = torch.cat(
            [
                cur.mean(dim=0),
                goal.mean(dim=0),
                res.abs().mean(dim=0),
            ],
            dim=-1,
        )  # (4D,)

        hidden = self.project(stats)
        return self.tail(hidden).reshape(1)  # (1,)
