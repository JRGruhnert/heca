from dataclasses import dataclass

import torch

from hoopgn.environments.properties.features.modifiers.modifier import (
    PropertyModifier,
)


class SkipModifier(PropertyModifier):
    @dataclass(kw_only=True)
    class Config(PropertyModifier.Config):
        pass

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x
