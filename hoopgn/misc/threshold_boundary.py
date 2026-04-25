from dataclasses import dataclass
from functools import cached_property
import torch

from hoopgn.entities.properties.normalizers.boundary import BoundaryNormalizer


class BoundaryThreshold:
    @dataclass(kw_only=True)
    class Config:
        boundary: BoundaryNormalizer.Config
        threshold: float = 0.05

    def __init__(
        self,
        cfg: Config,
    ):
        self.cfg = cfg
        self.boundary = BoundaryNormalizer.from_config(cfg.boundary)

    @cached_property
    def relative(self) -> torch.Tensor:
        return self.cfg.threshold * (self.boundary.lower - self.boundary.upper)
