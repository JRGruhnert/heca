from dataclasses import dataclass, field

import torch

from hoopgn.properties.normalizers.normalizer import (
    PropertyNormalizer,
)


class BoundaryNormalizer(PropertyNormalizer):
    @dataclass(kw_only=True)
    class Config(PropertyNormalizer.Config):
        lower: list[float]
        upper: list[float]

    def __init__(
        self,
        cfg: Config,
    ):
        super().__init__(cfg)
        self.cfg = cfg
        self.lower = torch.tensor(cfg.lower, dtype=torch.float32)
        self.upper = torch.tensor(cfg.upper, dtype=torch.float32)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        cx = torch.clamp(x, self.lower, self.upper)
        return (cx - self.lower) / (self.upper - self.lower)
