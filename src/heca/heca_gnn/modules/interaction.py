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

        self.combine = nn.Sequential(
            nn.Linear(4 * dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
        )

    def forward(self, option_x, gate_mask):
        good_idx = gate_mask.nonzero(as_tuple=True)[0]
        bad_idx = (~gate_mask).nonzero(as_tuple=True)[0]
        good = option_x[:, good_idx]  # (1, n_good, dim)
        bad = option_x[:, bad_idx]  # (1, n_bad, dim)

        all_att, _ = self.self_attn(option_x, option_x, option_x)
        all = self.norm1(option_x + all_att)
        all = all + self.ffn_all(all)

        good_att, _ = self.cross_good(good, option_x, option_x)
        good = self.norm2(good + good_att)
        good = good + self.ffn_good(good)

        bad_att, _ = self.cross_bad(bad, option_x, option_x)
        bad = self.norm3(bad + bad_att)
        bad = bad + self.ffn_bad(bad)

        good_out = torch.zeros_like(option_x)
        bad_out = torch.zeros_like(option_x)

        good_out[:, good_idx] = good
        bad_out[:, bad_idx] = bad

        combined = self.combine(
            torch.cat(
                [option_x, all, good_out, bad_out],
                dim=-1,
            )
        )

        return option_x + combined


class IdentityBlock(nn.Module):

    def forward(self, x: torch.Tensor, good_mask) -> torch.Tensor:
        return x


# good scenes: 0, 1, 8
# bad scenes: 2, 4
# fixable: 3, 5, 6, 7, 9, 10
