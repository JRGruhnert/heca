from torch import nn
import torch


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.cross_good = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.cross_bad = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, option_x, gate_mask):
        all_att, _ = self.self_attn(option_x, option_x, option_x)
        option_x = self.norm1(option_x + all_att)

        good_idx = gate_mask.nonzero(as_tuple=True)[0]
        bad_idx = (~gate_mask).nonzero(as_tuple=True)[0]

        good = option_x[:, good_idx]  # (1, n_good, dim)
        bad = option_x[:, bad_idx]  # (1, n_bad, dim)

        good_att, _ = self.cross_good(good, option_x, option_x)
        good = self.norm2(good + good_att)

        bad_att, _ = self.cross_bad(bad, option_x, option_x)
        bad = self.norm3(bad + bad_att)

        # pool each group to a fixed-size vector
        good_vec = good.mean(dim=1)  # (1, dim)
        bad_vec = bad.mean(dim=1)  # (1, dim)

        return torch.cat([good_vec, bad_vec], dim=-1)  # (1, 2*dim)


class IdentityBlock(nn.Module):

    def forward(self, x: torch.Tensor, good_mask) -> torch.Tensor:
        return x


# good scenes: 0, 1, 8
# bad scenes: 2, 4
# fixable: 3, 5, 6, 7, 9, 10
