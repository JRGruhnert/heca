from dataclasses import dataclass
import torch

from src.objects.properties.area import Area, AreaConfig
from src.objects.properties.value_handler.validators.validator import (
    Validator,
    ValidatorConfig,
)


@dataclass
class AreaValidatorConfig(ValidatorConfig):
    area: AreaConfig


class AreaValidator(Validator):
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
