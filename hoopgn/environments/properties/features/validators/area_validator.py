from dataclasses import dataclass
import torch

from hoopgn.environments.properties.v1.area_state import (
    AreaState,
)
from hoopgn.environments.properties.features.validators.validator import (
    PropertyValidator,
)


@dataclass(kw_only=True)
class AreaValidatorConfig(PropertyValidator.Config):
    area: AreaState.Config


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
