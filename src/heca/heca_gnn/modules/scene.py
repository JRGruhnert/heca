import torch
from torch import nn
from torch_geometric.nn import SAGEConv, GATv2Conv


class SceneGATBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = GATv2Conv(in_channels=(dim, dim), out_channels=dim)

    def forward(
        self,
        x_option: torch.Tensor,
        x_state: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        return self.conv((x_option, x_state), edge_index)


class SceneSageBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = SAGEConv(in_channels=(dim, dim), out_channels=dim, aggr="mean")
        self.head = nn.Sequential(
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(
        self,
        x_option: torch.Tensor,
        x_state: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        return self.head(self.conv((x_option, x_state), edge_index))
