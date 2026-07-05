from dataclasses import dataclass
from pathlib import Path

from torch import nn
import torch

from heca.misc.base import Configurable
from heca.heca_gnn.mlp import StandardMLP
from heca.misc import hardware


class PropertyEncoder(Configurable, nn.Module):
    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        subroot: str = "heca"
        label: str = "encoder"
        in_dim: int = 16
        out_dim: int = 32
        hidden_dim: int | None = None

    def __init__(self, cfg: Config):
        nn.Module.__init__(self)
        self.cfg = cfg
        self.fc = StandardMLP(cfg.in_dim, cfg.out_dim, cfg.hidden_dim)

    def forward(self, x):
        return self.fc(x)
