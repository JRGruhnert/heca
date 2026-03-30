from dataclasses import dataclass, field

import torch


@dataclass
class BoundaryConfig:
    lower: list[float]
    upper: list[float]


@dataclass
class AreaBoundaryConfig(BoundaryConfig):
    lower: list[float] = field(default_factory=lambda: [-1.0, -1.0, -1.0])
    upper: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])


@dataclass
class FlipBoundaryConfig(BoundaryConfig):
    lower: list[float] = field(default_factory=lambda: [0.0])
    upper: list[float] = field(default_factory=lambda: [1.0])


class Boundary:
    def __init__(
        self,
        config: BoundaryConfig,
    ):
        self.config = config
        self.lower = torch.tensor(config.lower, dtype=torch.float32)
        self.upper = torch.tensor(config.upper, dtype=torch.float32)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        cx = torch.clamp(x, self.lower, self.upper)
        return (cx - self.lower) / (self.upper - self.lower)
