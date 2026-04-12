from dataclasses import dataclass

import torch

from hoopgn.properties.features.modifiers.modifier import (
    PropertyModifier,
    PropertyModifierConfig,
)
from hoopgn.properties.states import select_state_property
from hoopgn.properties.state import StateConfig


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
