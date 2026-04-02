from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch

from src.states import state
from src.states.logic import area
from src.states.logic.area import Area, AreaConfig
from src.states.logic.value_handler.validators.validator import (
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

    @abstractmethod
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> bool:
        ax = self.area.label(x)
        ay = self.area.label(y)
        return ax is not None and ax == ay
