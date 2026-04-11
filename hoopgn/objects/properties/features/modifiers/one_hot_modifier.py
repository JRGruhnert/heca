from dataclasses import dataclass

import torch

from hoopgn.objects.properties.features.modifiers.modifier import (
    PropertyModifier,
    PropertyModifierConfig,
)
from hoopgn.objects.properties.states import select_state_property
from hoopgn.objects.properties.states.state import StateConfig


@dataclass(kw_only=True)
class OneHotModifierConfig(PropertyModifierConfig):
    state: StateConfig


class OneHotModifier(PropertyModifier):
    def __init__(
        self,
        config: OneHotModifierConfig,
    ):
        super().__init__(config)
        self.config = config
        self.state = select_state_property(config.state)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        lx = self.state(x)
        return torch.cat([x, lx], dim=0)
