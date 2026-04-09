from dataclasses import dataclass, field

import torch

from src.objects.properties.value_handler.normalizers.normalizer import (
    Normalizer,
    NormalizerConfig,
)


@dataclass
class BoundaryNormalizerConfig(NormalizerConfig):
    lower: list[float]
    upper: list[float]


@dataclass
class AreaBoundaryConfig(BoundaryNormalizerConfig):
    lower: list[float] = field(default_factory=lambda: [-1.0, -1.0, -1.0])
    upper: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])


@dataclass
class BoolBoundaryConfig(BoundaryNormalizerConfig):
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
