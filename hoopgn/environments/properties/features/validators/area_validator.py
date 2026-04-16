from dataclasses import dataclass
import torch

from hoopgn.environments.properties.states.area_state import AreaState, AreaStateConfig
from hoopgn.environments.properties.features.validators.validator import (
    PropertyValidator,
    PropertyValidatorConfig,
)


@dataclass(kw_only=True)
class AreaValidatorConfig(PropertyValidatorConfig):
    area: AreaStateConfig


class AreaValidator(PropertyValidator):
    def __init__(
        self,
        config: AreaValidatorConfig,
    ):
        self.config = config
        self.area = AreaState(config.area)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> bool:
        ax = self.area.label(x)
        ay = self.area.label(y)
        return ax is not None and ax == ay
