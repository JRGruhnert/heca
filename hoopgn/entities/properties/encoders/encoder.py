from dataclasses import dataclass

from torch import nn

from hoopgn.misc.classes import StoragableClass
from hoopgn.misc.mlp import StandardMLP
from hoopgn.misc.td import TDEntity


class PropertyEncoder(StoragableClass, nn.Module):
    @dataclass(kw_only=True)
    class Query(StoragableClass.Query):
        label: str

    @dataclass(kw_only=True)
    class Config(StoragableClass.Config):
        in_dim: int
        out_dim: int = 32
        hidden_dim: int | None = None

    def __init__(self, cfg: Config):
        nn.Module.__init__(self)
        self.cfg = cfg
        self.fc = StandardMLP(cfg.in_dim, cfg.out_dim, cfg.hidden_dim)

    def forward(self, x):
        return self.fc(x)
