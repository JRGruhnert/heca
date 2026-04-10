from dataclasses import dataclass
from functools import cached_property
import torch

from src.objects.properties.handlers.normalizers.boundary_normalizer import (
    BoundaryNormalizer,
    BoundaryNormalizerConfig,
)


@dataclass(kw_only=True)
class BoundaryThresholdConfig:
    boundary: BoundaryNormalizerConfig
    threshold: float = 0.05


class BoundaryThreshold:
    def __init__(
        self,
        config: BoundaryThresholdConfig,
    ):
        self.config = config
        self.boundary = BoundaryNormalizer(config.boundary)

    @cached_property
    def relative(self) -> torch.Tensor:
        """Returns the relative threshold for the state."""
        return self.config.threshold * (self.boundary.lower - self.boundary.upper)
