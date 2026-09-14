import torch
from torch import nn
from torch_geometric.nn import GINEConv, GATv2Conv

from heca.heca_gnn.modules.common import _make_gnn_mlp


class ConditionGinBlock(nn.Module):
    def __init__(self, dim: int, num_layers: int = 2):
        super().__init__()
        self.conv = GINEConv(nn=_make_gnn_mlp(dim, num_layers), edge_dim=8)

    def forward(
        self,
        x_comp: torch.Tensor,
        x_entity: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        return self.conv((x_comp, x_entity), edge_index, edge_attr) + x_entity


class ConditionGatBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = GATv2Conv(
            in_channels=dim,
            out_channels=dim // 4,
            heads=4,
            edge_dim=8,
        )

    def forward(
        self,
        x_comp: torch.Tensor,
        x_entity: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        return self.conv((x_comp, x_entity), edge_index, edge_attr) + x_entity
