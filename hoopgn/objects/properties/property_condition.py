from dataclasses import dataclass

import torch

from hoopgn.objects.properties.handlers.parameters import select_state_parameter
from hoopgn.objects.properties.handlers.parameters.ignore_parameter import (
    IgnoreParameterConfig,
)
from hoopgn.objects.properties.handlers.parameters.parameter import (
    StateParameterConfig,
)
from hoopgn.objects.properties.handlers.rulers import select_state_ruler
from hoopgn.objects.properties.handlers.rulers.ignore_ruler import IgnoreRulerConfig
from hoopgn.objects.properties.handlers.rulers.ruler import RulerConfig


@dataclass(kw_only=True)
class PropertyConditionConfig:
    ruler: RulerConfig
    parameter: StateParameterConfig
    value: list[float] | float | int | None = None
    last_digit: int = 0


@dataclass
class IgnoreConditionConfig(PropertyConditionConfig):
    ruler: RulerConfig = IgnoreRulerConfig()
    parameter: StateParameterConfig = IgnoreParameterConfig()
    value: list[float] | float | int | None = None
    last_digit: int = 1


class PropertyCondition:
    def __init__(self, config: PropertyConditionConfig):
        self.config = config
        self.ruler = select_state_ruler(config.ruler)
        self.parameter = select_state_parameter(config.parameter)
        self.value = self.parameter(config.value)
        assert self.value is not None, f"No {type(self)} value."

    def __call__(self, x: torch.Tensor, y: torch.Tensor | None = None) -> float:
        if y is not None:
            return self.ruler(x, y)
        return self.ruler(x, self.value)

    def edge_feature(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.tensor(
            [self.ruler(a, b), self.config.last_digit], dtype=torch.float32
        )

    @classmethod
    def from_demos(
        cls, value: tuple, config: PropertyConditionConfig
    ) -> "PropertyCondition":
        instance = cls(config)
        start, end, reversed, selected_by_tapas = value
        tp = instance.parameter.hoopgnv1(start, end, reversed, selected_by_tapas)

        if tp is None:
            return PropertyCondition(IgnoreConditionConfig())

        instance.value = tp
        return instance
