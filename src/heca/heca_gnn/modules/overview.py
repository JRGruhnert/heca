import torch
from torch import nn
from torch_geometric.nn import GINEConv
from heca.heca_gnn.modules.common import _make_gnn_mlp


class SceneGNNBlock(nn.Module):
    def __init__(self, dim: int, num_layers: int = 2):
        super().__init__()
        self.conv = GINEConv(nn=_make_gnn_mlp(dim, num_layers))

    def forward(
        self,
        x_option: torch.Tensor,
        x_state: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        return self.conv((x_option, x_state), edge_index, edge_attr)


class SceneTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.cross_good = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.cross_bad = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, all_opts, good_mask):
        all_att, _ = self.self_attn(all_opts, all_opts, all_opts)
        all_opts = self.norm1(all_opts + all_att)

        good_idx = good_mask.nonzero(as_tuple=True)[0]
        bad_idx = (~good_mask).nonzero(as_tuple=True)[0]

        good = all_opts[:, good_idx]  # (1, n_good, dim)
        bad = all_opts[:, bad_idx]  # (1, n_bad, dim)

        good_att, _ = self.cross_good(good, all_opts, all_opts)
        good = self.norm2(good + good_att)

        bad_att, _ = self.cross_bad(bad, all_opts, all_opts)
        bad = self.norm3(bad + bad_att)

        # pool each group to a fixed-size vector
        good_vec = good.mean(dim=1)  # (1, dim)
        bad_vec = bad.mean(dim=1)  # (1, dim)

        return torch.cat([good_vec, bad_vec], dim=-1)  # (1, 2*dim)
