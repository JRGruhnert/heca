from typing import Sequence

import torch
from torch import nn
from torch_geometric.nn import PairNorm


class BoundaryNorm(nn.Module):
    def __init__(self, dim: int, names: Sequence[str]):
        nn.Module.__init__(self)
        self.names = tuple(names)
        self.norms = nn.ModuleDict({name: nn.LayerNorm(dim) for name in names})

    def forward(self, name: str, x: torch.Tensor) -> torch.Tensor:
        if name not in self.norms:
            raise KeyError(
                f"unknown boundary {name!r}; this segment has {list(self.names)}"
            )
        return self.norms[name](x)


class PairNormBlock(nn.Module):
    def __init__(self, streams: Sequence[str], enabled: bool):
        nn.Module.__init__(self)
        self.enabled = enabled
        self.norms = (
            nn.ModuleDict({name: PairNorm() for name in streams})
            if enabled
            else nn.ModuleDict()
        )

    def forward(self, stream: str, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        return self.norms[stream](x)


class NoUpdateBlock(nn.Module):
    def forward(self, x: torch.Tensor, *rest: object) -> torch.Tensor:
        return torch.zeros_like(x)
