from dataclasses import dataclass

import torch

from hoopgn.objects.properties.features.modifiers.modifier import (
    Modifier,
    ModifierConfig,
)


@dataclass(kw_only=True)
class SkipModifierConfig(ModifierConfig):
    pass


class SkipModifier(Modifier):
    def __init__(
        self,
        config: SkipModifierConfig,
    ):
        super().__init__(config)
        self.config = config

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x
