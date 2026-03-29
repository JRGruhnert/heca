from dataclasses import dataclass

import torch

from src.states.logic.boundary import Boundary, BoundaryConfig
from src.states.logic.value_cnd import ValueCondition, ValueConditionConfig


@dataclass
class LinearValueNormalizerConfig(ValueConditionConfig):
    boundary: BoundaryConfig = BoundaryConfig(
        lower_bound=[0.0],
        upper_bound=[1.0],
    )


class LinearValueNormalizer(ValueCondition):
    def __init__(
        self,
        config: LinearValueNormalizerConfig,
    ):
        super().__init__(config)
        self.boundary = Boundary(config.boundary)

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """Clamp and normalize the input value."""
        return self.boundary.normalize(x)

    def make_input(self, x: torch.Tensor) -> torch.Tensor:
        """Clamp and normalize the input value."""
        return self.value(x)
