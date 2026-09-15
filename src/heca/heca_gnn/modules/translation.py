import torch
from torch import nn


class TranslationBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        assert dst.unique().numel() == dst.numel(), (
            "translation edges must be 1-to-1 into dst; use a scatter for "
            "multi-predecessor rows"
        )
        msg = self.mlp(torch.cat([x[src], x[dst]], dim=-1))
        return x.index_copy(0, dst, msg)
