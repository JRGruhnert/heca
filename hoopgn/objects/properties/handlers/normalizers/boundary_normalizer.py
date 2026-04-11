from dataclasses import dataclass, field

import torch

from hoopgn.objects.properties.handlers.normalizers.normalizer import (
    Normalizer,
    NormalizerConfig,
)


@dataclass(kw_only=True)
class BoundaryNormalizerConfig(NormalizerConfig):
    lower: list[float]
    upper: list[float]


@dataclass(kw_only=True)
class AreaNormalizerConfig(BoundaryNormalizerConfig):
    lower: list[float] = field(default_factory=lambda: [-1.0, -1.0, -1.0])
    upper: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])


@dataclass(kw_only=True)
class BoolNormalizerConfig(BoundaryNormalizerConfig):
    lower: list[float] = field(default_factory=lambda: [0.0])
    upper: list[float] = field(default_factory=lambda: [1.0])


class BoundaryNormalizer(Normalizer):
    def __init__(
        self,
        config: BoundaryNormalizerConfig,
    ):
        self.config = config
        self.lower = torch.tensor(config.lower, dtype=torch.float32)
        self.upper = torch.tensor(config.upper, dtype=torch.float32)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        cx = torch.clamp(x, self.lower, self.upper)
        return (cx - self.lower) / (self.upper - self.lower)
