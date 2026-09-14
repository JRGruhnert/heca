import torch
from torch import nn
from torch_geometric.nn import GINConv, GCN2Conv
from heca.heca_gnn.modules.common import _make_gnn_mlp


class SummaryGinBlock(nn.Module):
    def __init__(self, dim: int, num_layers: int = 2):
        super().__init__()
        self.conv = GINConv(nn=_make_gnn_mlp(dim, num_layers))

    def forward(
        self,
        x_entity: torch.Tensor,
        x_option: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        return self.conv((x_entity, x_option), edge_index)


class SummaryGcnBlock(nn.Module):
    def __init__(self, dim: int, alpha: float = 0.5):
        super().__init__()
        self.conv = GCN2Conv(channels=dim, alpha=alpha)

    def forward(
        self,
        x_entity: torch.Tensor,
        x_option: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        return self.conv((x_entity, x_option), edge_index)
