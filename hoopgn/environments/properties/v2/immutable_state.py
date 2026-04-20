from dataclasses import dataclass, field
import torch

from hoopgn.environments.properties.state import State


class LockedState(State):
    @dataclass(kw_only=True)
    class Config(State.Config):
        values: set[str] = field(default_factory=lambda: {"None"})

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg

    def label(self, x: torch.Tensor) -> str:
        return "None"
