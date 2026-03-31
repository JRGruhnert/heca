from dataclasses import dataclass

import torch

from src.states.logic.boundary import Boundary, BoundaryConfig
from src.states.logic.values.value import Value, ValueHandlerConfig


@dataclass
class LinearValueConfig(ValueHandlerConfig):
    boundary: BoundaryConfig = BoundaryConfig(
        lower=[0.0],
        upper=[1.0],
    )


class LinearValue(Value):
    def __init__(
        self,
        config: LinearValueConfig,
    ):
        super().__init__(config)
        self.boundary = Boundary(config.boundary)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Clamp and normalize the input value."""
        return self.boundary.normalize(x)

    def make_input(self, x: torch.Tensor) -> torch.Tensor:
        """Clamp and normalize the input value."""
        return self.__call__(x)
