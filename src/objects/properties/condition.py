from dataclasses import dataclass

import torch

from src.factory import select_state_ruler, select_state_parameter
from src.objects.properties.value_handler.parameters.ignore_parameter import (
    IgnoreParameterConfig,
)
from src.objects.properties.value_handler.parameters.parameter import (
    StateParameterConfig,
)
from src.objects.properties.value_handler.rulers.ignore_ruler import IgnoreRulerConfig
from src.objects.properties.value_handler.rulers.ruler import RulerConfig


@dataclass
class ConditionConfig:
    ruler: RulerConfig
    parameter: StateParameterConfig
    value: list[float] | float | int | None = None


@dataclass
class IgnoreConditionConfig(ConditionConfig):
    ruler: RulerConfig = IgnoreRulerConfig()
    parameter: StateParameterConfig = IgnoreParameterConfig()
    value: list[float] | float | int | None = None


class Condition:
    def __init__(self, config: ConditionConfig):
        self.config = config
        self.ruler = select_state_ruler(config.ruler)
        self.parameter = select_state_parameter(config.parameter)
        self.value = self.parameter(config.value)

    def __call__(self, x: torch.Tensor, y: torch.Tensor | None = None) -> float:
        if y is not None:
            return self.ruler(x, y)
        return self.ruler(x, self.value)

    def edge_feature(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.ruler.edge_feature(a, b)

    @classmethod
    def from_demos(cls, value: tuple, config: ConditionConfig) -> "Condition":
        instance = cls(config)
        start, end, reversed, selected_by_tapas = value
        tp = instance.parameter.hoopgnv1(start, end, reversed, selected_by_tapas)

        if tp is None:
            return Condition(IgnoreConditionConfig())

        instance.value = tp
        return instance
