from dataclasses import dataclass

import torch

from hoopgn.properties.features.parameters.parameter import (
    PropertyParameter,
)

from hoopgn.properties.features.rulers.ruler import PropertyRuler
from hoopgn.properties.features.feature import PropertyFeature


class PropertyCondition(PropertyFeature):
    @dataclass(kw_only=True)
    class Config(PropertyFeature.Config):
        ruler: PropertyRuler.Config
        parameter: PropertyParameter.Config
        value: list[float] | float | int | None = None
        last_digit: int = 0

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.ruler = PropertyRuler.from_config(cfg.ruler)
        self.parameter = PropertyParameter.from_config(cfg.parameter)
        self.value = self.parameter(cfg.value)
        assert self.value is not None, f"No {type(self)} value."

    def __call__(self, x: torch.Tensor, y: torch.Tensor | None = None) -> float:
        if y is not None:
            return self.ruler(x, y)
        return self.ruler(x, self.value)

    def edge_feature(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.tensor(
            [self.ruler(a, b), self.cfg.last_digit], dtype=torch.float32
        )
