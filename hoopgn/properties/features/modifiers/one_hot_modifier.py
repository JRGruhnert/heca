from dataclasses import dataclass

import torch

from hoopgn.entities.features.modifiers.modifier import (
    PropertyModifier,
)
from hoopgn.properties.features.modifiers.modifier import (
    PropertyModifier,
)
from hoopgn.properties.states.state import State


class OneHotModifier(PropertyModifier):
    @dataclass(kw_only=True)
    class Config(PropertyModifier.Config):
        state: State.Config

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.state = State.from_config(cfg.state)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        lx = self.state(x)
        return torch.cat([x, lx], dim=0)
