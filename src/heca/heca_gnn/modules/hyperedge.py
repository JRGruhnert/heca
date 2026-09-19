import torch
from torch import nn

from heca.graphs.roles import ENRole


class TypedHyperedgeLayer(nn.Module):
    def __init__(self, dim: int, num_roles: int):
        super().__init__()
        self.role_emb = nn.Embedding(num_roles, dim)  # global role table
        self.out = nn.Linear(dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        roles: torch.Tensor,
        entity_ids: torch.Tensor,
    ) -> torch.Tensor:
        goal_rows = (roles == ENRole.GOAL.value).nonzero(as_tuple=True)[0]
        if goal_rows.numel() == 0:
            return torch.zeros_like(x)
        h = x + self.role_emb(roles.clamp(min=0))
        member = entity_ids[goal_rows, None] == entity_ids[None]  # (G, n)
        member[:, goal_rows] = False  # a goal does not condition itself
        return member.t().float() @ self.out(h[goal_rows])  # (n, dim)
