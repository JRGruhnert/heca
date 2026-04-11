from dataclasses import dataclass

import torch

from hoopgn.objects.properties.features.modifiers.modifier import (
    PropertyModifier,
    PropertyModifierConfig,
)


@dataclass(kw_only=True)
class SkipModifierConfig(PropertyModifierConfig):
    pass


class SkipModifier(PropertyModifier):
    def __init__(
        self,
        config: SkipModifierConfig,
    ):
        super().__init__(config)
        self.config = config

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x
