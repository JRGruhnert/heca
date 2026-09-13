import torch
from torch import nn


class CriticNetwork(nn.Module):
    def __init__(self, dim: int, hidden_ratio: float = 0.5):
        super().__init__()
        hidden = max(int(dim * hidden_ratio), 16)

        self.project = nn.Linear(4 * dim, dim)

        self.tail = nn.Sequential(
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        entity_x: torch.Tensor,
        cur_idx: torch.Tensor,
        goal_idx: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        cur = entity_x[cur_idx]  # (E, D) — same entity order as goal
        goal = entity_x[goal_idx]  # (E, D)
        res = cur - goal  # per-entity residual (progress toward the goal)

        # per-entity progress toward the goal + the pooled situation (1, D)
        stats = torch.cat(
            [
                cur.mean(dim=0),
                goal.mean(dim=0),
                res.abs().mean(dim=0),
                state.reshape(-1),
            ],
            dim=-1,
        )  # (4D,)

        hidden = self.project(stats)
        return self.tail(hidden).reshape(1)  # (1,)
