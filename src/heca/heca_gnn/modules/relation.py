import torch
from torch import nn
from torch_geometric.nn import GINConv


class OptionRelationBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = GINConv(
            nn=nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
            ),
        )

    def forward(self, x_option: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            # no option pair shares an entity: keep the rows unchanged
            return x_option.new_zeros(x_option.shape)
        return self.conv((x_option, x_option), edge_index)
