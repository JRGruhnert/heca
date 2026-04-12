from dataclasses import dataclass

import torch

from hoopgn.properties.features.parameters import select_property_parameter
from hoopgn.properties.features.parameters.ignore_parameter import (
    IgnoreParameterConfig,
)
from hoopgn.properties.features.parameters.parameter import (
    PropertyParameterConfig,
)
from hoopgn.properties.features.rulers import select_property_ruler
from hoopgn.properties.features.rulers.ignore_ruler import IgnoreRulerConfig
from hoopgn.properties.features.rulers.ruler import PropertyRulerConfig
from hoopgn.properties.features.feature import (
    PropertyFeature,
    PropertyFeatureConfig,
)


@dataclass(kw_only=True)
class PropertyConditionConfig(PropertyFeatureConfig):
    ruler: PropertyRulerConfig
    parameter: PropertyParameterConfig
    value: list[float] | float | int | None = None
    last_digit: int = 0


class PropertyCondition(PropertyFeature):
    def __init__(self, config: PropertyConditionConfig):
        super().__init__(config)
        self.config = config
        self.ruler = select_property_ruler(config.ruler)
        self.parameter = select_property_parameter(config.parameter)
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


@dataclass(kw_only=True)
class IgnoreConditionConfig(PropertyConditionConfig):
    ruler: IgnoreRulerConfig = IgnoreRulerConfig()
    parameter: IgnoreParameterConfig = IgnoreParameterConfig()
    value: list[float] | float | int | None = None
    last_digit: int = 1
