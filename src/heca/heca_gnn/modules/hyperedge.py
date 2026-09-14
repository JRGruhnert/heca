from torch import nn


class TypedHyperedgeLayer(nn.Module):

    def __init__(self, dim: int, num_roles: int, num_heads: int = 4):
        super().__init__()
        self.role_emb = nn.Embedding(num_roles, dim)  # global role table
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, roles):
        # x:     (n_in_hyperedge, dim)
        # roles: (n_in_hyperedge,) int, global role ids
        h = x + self.role_emb(roles)
        hs = h.unsqueeze(0)
        att, _ = self.attn(hs, hs, hs)
        return self.norm(hs.squeeze(0) + att.squeeze(0))
