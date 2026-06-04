from dataclasses import dataclass

import torch
from torch import nn

from heca.misc.base import Configurable


class BaseNetwork(Configurable, nn.Module):
    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        feature_dim: int = 32

    def __init__(self, cfg: Config):
        nn.Module.__init__(self)
        self.cfg = cfg

    def forward(
        self, x: torch.Tensor, x_dict: dict, edge_index_dict: dict, edge_attr_dict: dict
    ) -> torch.Tensor:
        raise NotImplementedError()
