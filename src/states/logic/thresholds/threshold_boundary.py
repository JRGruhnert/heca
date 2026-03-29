from functools import cached_property
import torch

from src.states.logic.boundary import Boundary, BoundaryConfig


class BoundaryThresholdConfig:
    boundary: BoundaryConfig
    threshold: float = 0.05


class BoundaryThreshold:
    def __init__(
        self,
        config: BoundaryThresholdConfig,
    ):
        self.config = config
        self.boundary = Boundary(config.boundary)

    @cached_property
    def relative(self) -> torch.Tensor:
        """Returns the relative threshold for the state."""
        return self.config.threshold * (
            self.boundary.max_limit - self.boundary.lower_limit
        )

    @cached_property
    def absolute(self) -> torch.Tensor:
        """Returns the absolute threshold for the state."""
        return self.relative
