from torch import nn
import torch


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, ff_dim=None):
        super().__init__()

        ff_dim = ff_dim or 4 * dim

        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.cross_good = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.cross_bad = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        self.ffn_all = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, dim),
        )

        self.ffn_good = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, dim),
        )

        self.ffn_bad = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, dim),
        )

        self.combine = nn.Linear(4 * dim, dim)

    def forward(self, option_x: torch.Tensor, gate_mask: torch.Tensor) -> torch.Tensor:
        good_mask = ~gate_mask.bool()
        good = option_x[good_mask]  # (n_good, dim)
        bad = option_x[~good_mask]  # (n_bad, dim)

        query = option_x.unsqueeze(0)  # MultiheadAttention wants a batch dim
        all_att, _ = self.self_attn(query, query, query)
        all_x = self.norm1(option_x + all_att.squeeze(0))
        all_x = all_x + self.ffn_all(all_x)

        good_att, _ = self.cross_good(good.unsqueeze(0), query, query)
        good_x = self.norm2(good + good_att.squeeze(0))
        good_x = good_x + self.ffn_good(good_x)

        bad_att, _ = self.cross_bad(bad.unsqueeze(0), query, query)
        bad_x = self.norm3(bad + bad_att.squeeze(0))
        bad_x = bad_x + self.ffn_bad(bad_x)

        good_out = torch.zeros_like(option_x)
        bad_out = torch.zeros_like(option_x)
        good_out[good_mask] = good_x
        bad_out[~good_mask] = bad_x

        return self.combine(torch.cat([option_x, all_x, good_out, bad_out], dim=-1))
