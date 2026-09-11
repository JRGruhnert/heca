from torch import nn
import torch


class StateCritic(nn.Module):
    SITE = "critic"

    def __init__(self, dim: int, use_budget: bool = True, hidden_ratio: float = 0.5):
        super().__init__()
        hidden = max(int(dim * hidden_ratio), 16)
        self.use_budget = use_budget

        self.row_net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

        self.project = nn.Linear(4 * dim + int(use_budget), dim)
        self.norm = nn.LayerNorm(dim)
        self.tail = nn.Sequential(
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
        films,
        conds: dict,
        budget: torch.Tensor,
    ) -> torch.Tensor:
        cur = canonical_x[cur_idx]  # (E, D) — same entity order as goal
        goal = canonical_x[goal_idx]  # (E, D)
        res = cur - goal  # per-entity residual (progress toward the goal)

        stats = torch.cat(
            [
                cur.mean(dim=0),
                goal.mean(dim=0),
                res.abs().mean(dim=0),
                self.row_net(res).mean(dim=0),
            ],
            dim=-1,
        )  # (4D,)
        if self.use_budget:
            stats = torch.cat([stats, budget.reshape(1)], dim=-1)  # (4D + 1,)
        self.last_stats = stats.detach()

        hidden = films(self.norm(self.project(stats)), conds, self.SITE)
        return self.tail(hidden).reshape(1)  # (1,)
