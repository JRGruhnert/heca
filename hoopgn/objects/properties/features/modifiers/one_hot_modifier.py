from dataclasses import dataclass

import torch

from hoopgn.objects.properties.features.modifiers.modifier import (
    Modifier,
    ModifierConfig,
)


from hoopgn.objects.properties.x_state import XState, XStateConfig


@dataclass(kw_only=True)
class OneHotModifierConfig(ModifierConfig):
    state: XStateConfig


class OneHotModifier(Modifier):
    def __init__(
        self,
        config: OneHotModifierConfig,
    ):
        super().__init__(config)
        self.state = XState(config.state)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        lx = self.state(x)
        return torch.cat([x, lx], dim=0)
