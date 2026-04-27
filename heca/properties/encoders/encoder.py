from dataclasses import dataclass

from torch import nn

from heca.misc.classes import Persistable
from heca.heca_gnn.mlp import StandardMLP


class PropertyEncoder(Persistable, nn.Module):
    @dataclass(kw_only=True)
    class Config(Persistable.Config):
        in_dim: int
        out_dim: int = 32
        hidden_dim: int | None = None

    def __init__(self, cfg: Config):
        nn.Module.__init__(self)
        self.cfg = cfg
        self.fc = StandardMLP(cfg.in_dim, cfg.out_dim, cfg.hidden_dim)

    def forward(self, x):
        return self.fc(x)
