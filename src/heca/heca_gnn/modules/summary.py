import torch
from torch import nn
from torch_geometric.nn import GINConv, SAGEConv
import torch.nn.functional as F


class SummaryGinBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = GINConv(
            nn=nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
            ),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x_entity: torch.Tensor,
        x_option: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        x = self.conv((x_entity, x_option), edge_index)
        x = self.norm(x)
        return F.gelu(x)


class SummarySageBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = SAGEConv(
            in_channels=(dim, dim),
            out_channels=dim,
            aggr="mean",
        )
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x_entity: torch.Tensor,
        x_option: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        x = self.conv((x_entity, x_option), edge_index)
        x = self.norm(x)
        return F.gelu(x)
