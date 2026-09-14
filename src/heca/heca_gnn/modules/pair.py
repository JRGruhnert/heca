import torch
from torch import nn

from heca.heca_gnn.modules.common import _make_gnn_mlp


class PairBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_layers: int = 2,
    ):
        super().__init__()
        self.mlp = _make_gnn_mlp(dim, num_layers, in_dim=2 * dim)

    def forward(
        self,
        x_src: torch.Tensor,
        x_dst: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index
        return self.mlp(torch.cat([x_src[src], x_dst[dst]], dim=-1))
