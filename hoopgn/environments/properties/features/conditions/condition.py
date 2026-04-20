from dataclasses import dataclass

import torch

from hoopgn import logger
from hoopgn.environments.properties.features.parameters.ignore_parameter import (
    IgnoreParameter,
)
from hoopgn.environments.properties.features.parameters.parameter import (
    PropertyParameter,
)

from hoopgn.environments.properties.features.rulers.ignore_ruler import IgnoreRuler
from hoopgn.environments.properties.features.rulers.ruler import PropertyRuler
from hoopgn.environments.properties.features.feature import PropertyFeature
from hoopgn.environments.properties.property import Property


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

    @classmethod
    def from_hoopgnv1_demos(
        cls,
        value: tuple,
        config: Property.Config,
    ) -> "PropertyCondition":
        instance = cls(config.condition)
        start, end, reversed, selected_by_tapas = value
        tp = instance.parameter.hoopgnv1(start, end, reversed, selected_by_tapas)
        if tp is None:
            return PropertyCondition(
                PropertyCondition.Config(
                    label=config.label,
                    ruler=IgnoreRuler.Config(
                        label="ignore",
                    ),
                    parameter=IgnoreParameter.Config(
                        label="ignore",
                    ),
                    value=None,
                    last_digit=1,
                )
            )

        instance.value = tp
        logger.debug(f"Created {cls.__name__} from demos with value: {instance.value}.")
        return instance
