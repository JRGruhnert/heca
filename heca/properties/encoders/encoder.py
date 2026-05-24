from dataclasses import dataclass
from pathlib import Path

from torch import nn
import torch

from heca.classes.persist import Persistable
from heca.heca_gnn.mlp import StandardMLP
from heca.misc import hardware


class PropertyEncoder(Persistable, nn.Module):

    @dataclass(frozen=True, kw_only=True)
    class Query(Persistable.Query):
        label: str

    @dataclass(kw_only=True)
    class Config(Persistable.Config):
        in_dim: int = 16
        out_dim: int = 32
        hidden_dim: int | None = None

    @dataclass(frozen=True, kw_only=True)
    class Location(Persistable.Location):
        pass

    def __init__(self, cfg: Config):
        nn.Module.__init__(self)
        self.cfg = cfg
        self.fc = StandardMLP(cfg.in_dim, cfg.out_dim, cfg.hidden_dim)

    def forward(self, x):
        return self.fc(x)

    def _load(self, path: Path):
        self.load_state_dict(
            torch.load(path / "model.pt", map_location=hardware.device)
        )

    def _save(self, path: Path) -> bool:
        torch.save(self.state_dict(), path / "model.pt")
        return True
