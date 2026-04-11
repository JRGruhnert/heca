from dataclasses import dataclass
import torch

from hoopgn.objects.properties.area import Area, AreaConfig
from hoopgn.objects.properties.features.validators.validator import (
    PropertyValidator,
    PropertyValidatorConfig,
)


@dataclass(kw_only=True)
class AreaValidatorConfig(PropertyValidatorConfig):
    area: AreaConfig


class AreaValidator(PropertyValidator):
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
