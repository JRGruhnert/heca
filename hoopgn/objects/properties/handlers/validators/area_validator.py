from dataclasses import dataclass
import torch

from hoopgn.objects.properties.area import Area, AreaConfig
from hoopgn.objects.properties.handlers.validators.validator import (
    StateValidator,
    StateValidatorConfig,
)


@dataclass(kw_only=True)
class AreaValidatorConfig(StateValidatorConfig):
    area: AreaConfig


class AreaValidator(StateValidator):
    def __init__(
        self,
        config: AreaValidatorConfig,
    ):
        self.config = config
        self.area = Area(config.area)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> bool:
        ax = self.area.label(x)
        ay = self.area.label(y)
        return ax is not None and ax == ay
