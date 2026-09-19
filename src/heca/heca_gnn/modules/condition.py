import torch

from torch import nn
from torch_geometric.nn import GATv2Conv, GINEConv

from heca.graphs.edges.condition_edges import ConditionEdges
from heca.graphs.edges.edge_set import DEFAULT_TERMS


class ConditionGinBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        rotation: bool = True,
        terms: tuple[str, ...] = DEFAULT_TERMS,
        goal_residual: bool = False,
    ):
        super().__init__()
        self.conv = GINEConv(
            nn=nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
            ),
            edge_dim=ConditionEdges.edge_dim(rotation, terms, goal_residual),
        )

    def forward(
        self,
        x_comp: torch.Tensor,
        x_entity: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        return self.conv((x_comp, x_entity), edge_index, edge_attr)


class ConditionGatBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        rotation: bool = True,
        terms: tuple[str, ...] = DEFAULT_TERMS,
        goal_residual: bool = False,
    ):
        super().__init__()
        self.conv = GATv2Conv(
            in_channels=dim,
            out_channels=dim // 4,
            heads=4,
            edge_dim=ConditionEdges.edge_dim(rotation, terms, goal_residual),
            add_self_loops=False,
        )

    def forward(
        self,
        x_comp: torch.Tensor,
        x_entity: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        return self.conv((x_comp, x_entity), edge_index, edge_attr)
