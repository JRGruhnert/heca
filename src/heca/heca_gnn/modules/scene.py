import torch
from torch import nn
from torch_geometric.nn import GINEConv, SAGEConv, GATv2Conv
import torch.nn.functional as F


class SceneGATBlock(nn.Module):
    EDGE_DIM = 1

    def __init__(self, dim: int):
        super().__init__()
        self.conv = GATv2Conv(
            in_channels=(dim, dim), out_channels=dim, edge_dim=self.EDGE_DIM
        )
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x_option: torch.Tensor,
        x_state: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        x = self.conv((x_option, x_state), edge_index, edge_attr)
        x = self.norm(x)
        return F.gelu(x)


class SceneSageBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = SAGEConv(in_channels=(dim, dim), out_channels=dim, aggr="mean")
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x_option: torch.Tensor,
        x_state: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,  # Unused
    ) -> torch.Tensor:
        x = self.conv((x_option, x_state), edge_index)
        x = self.norm(x)
        return F.gelu(x)


class SceneGINBlock(nn.Module):
    EDGE_DIM = 1

    def __init__(self, dim: int):
        super().__init__()
        self.conv = GINEConv(
            nn=nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
            ),
            edge_dim=self.EDGE_DIM,
        )
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x_option: torch.Tensor,
        x_state: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        x = self.conv((x_option, x_state), edge_index, edge_attr)
        x = self.norm(x)
        return F.gelu(x)
