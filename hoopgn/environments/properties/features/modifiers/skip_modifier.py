from dataclasses import dataclass

import torch

from hoopgn.environments.properties.features.modifiers.modifier import PropertyModifier


class DefaultModifier(PropertyModifier):
    @dataclass(kw_only=True)
    class Config(PropertyModifier.Config):
        label: str = "default"

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """A default modifier that returns the input tensor unchanged"""
        return x
