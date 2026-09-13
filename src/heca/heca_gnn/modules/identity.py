from torch import nn
import torch


class IdentityBlock(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
