from dataclasses import dataclass

from torch import nn

from heca.classes.persist import Persistable
from heca.heca_gnn.mlp import StandardMLP


class PropertyEncoder(Persistable, nn.Module):

    @dataclass(frozen=True, kw_only=True)
    class Query(Persistable.Query):
        label: str

    @dataclass(kw_only=True)
    class Config(Persistable.Config):
        in_dim: int
        out_dim: int = 32
        hidden_dim: int | None = None

    @dataclass(frozen=True, kw_only=True)
    class File(Persistable.File):
        pass

    def __init__(self, cfg: Config):
        nn.Module.__init__(self)
        self.cfg = cfg
        self.fc = StandardMLP(cfg.in_dim, cfg.out_dim, cfg.hidden_dim)

    def forward(self, x):
        return self.fc(x)

    @classmethod
    def load(cls, query: "PropertyEncoder.Query") -> "PropertyEncoder":
        return cls(cls.Config(in_dim=16))

    # TODO: Implement loading from disk

    @classmethod
    def save(cls, query: "PropertyEncoder.Query") -> bool:
        return True

    # TODO: Implement saving to disk
