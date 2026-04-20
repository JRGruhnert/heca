from dataclasses import dataclass
from functools import cached_property
import torch

from hoopgn.properties.features.normalizers.boundary_normalizer import (
    BoundaryNormalizer,
    BoundaryNormalizerConfig,
)


class BoundaryThreshold:
    @dataclass(kw_only=True)
    class Config:
        boundary: BoundaryNormalizerConfig
        threshold: float = 0.05

    def __init__(
        self,
        cfg: Config,
    ):
        self.cfg = cfg
        self.boundary = BoundaryNormalizer(cfg.boundary)

    @cached_property
    def relative(self) -> torch.Tensor:
        """Returns the relative threshold for the state."""
        return self.cfg.threshold * (self.boundary.lower - self.boundary.upper)
