import torch
from torch import nn
from torch_geometric.nn import GINEConv, GATv2Conv
import torch.nn.functional as F


class ConditionGinBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = GINEConv(
            nn=nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
            ),
            edge_dim=8,
        )
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x_comp: torch.Tensor,
        x_entity: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        x = self.conv((x_comp, x_entity), edge_index, edge_attr) + x_entity
        x = self.norm(x)
        return F.gelu(x)


class ConditionGatBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = GATv2Conv(
            in_channels=dim,
            out_channels=dim // 4,
            heads=4,
            edge_dim=8,
        )
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x_comp: torch.Tensor,
        x_entity: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        x = self.conv((x_comp, x_entity), edge_index, edge_attr) + x_entity
        x = self.norm(x)
        return F.gelu(x)
