import torch
from torch import nn

from heca.heca_gnn.modules.pair import PairBlock


class TranslationBlock(nn.Module):

    def __init__(self, dim: int, num_layers: int = 2):
        super().__init__()
        self.pair = PairBlock(dim, num_layers=num_layers)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.pair(x, x, edge_index)
