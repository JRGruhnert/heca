import torch
from torch import nn

from heca.graphs.roles import ENRole


class TypedHyperedgeLayer(nn.Module):
    def __init__(self, dim: int, num_roles: int, num_heads: int = 4):
        super().__init__()
        self.role_emb = nn.Embedding(num_roles, dim)  # global role table
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.out = nn.Linear(dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        roles: torch.Tensor,
        type_ids: torch.Tensor,
    ) -> torch.Tensor:
        # x: (n, dim) entity rows, roles/type_ids: (n,) int
        update = torch.zeros_like(x)
        h = x + self.role_emb(roles.clamp(min=0))
        goal_rows = (roles == ENRole.GOAL.value).nonzero(as_tuple=True)[0]
        post_rows = (roles == ENRole.POST.value).nonzero(as_tuple=True)[0]
        if goal_rows.numel() == 0 or post_rows.numel() == 0:
            return update
        member = type_ids[goal_rows, None] == type_ids[None, post_rows]  # (G, P)
        keeps = member.any(dim=1)  # goals of a type that has no post row
        goal_rows, member = goal_rows[keeps], member[keeps]
        if goal_rows.numel() == 0:
            return update
        query, key = h[goal_rows][None], h[post_rows][None]  # (1, G/P, dim)
        msg, _ = self.attn(query, key, key, attn_mask=~member)
        residual = member.t().float() @ self.out(msg[0])  # (P, dim)
        return update.index_copy(0, post_rows, residual)
