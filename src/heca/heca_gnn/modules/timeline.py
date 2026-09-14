from torch import nn
import torch


class MemoryBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gru = nn.GRUCell(dim, dim)

    def forward(self, u: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        if h is None:
            h = u.new_zeros(u.shape[0], self.gru.hidden_size)
        return self.gru(u, h)


class NoMemoryBlock(nn.Module):

    def forward(self, x: torch.Tensor, h: torch.Tensor | None) -> torch.Tensor | None:
        return None
