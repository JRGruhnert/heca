from dataclasses import dataclass

import torch


@dataclass
class BoundaryConfig:
    lower_bound: list[float]
    upper_bound: list[float]


class Boundary:
    """Mixin for success conditions that need bounds"""

    def __init__(
        self,
        config: BoundaryConfig,
    ):
        self.config = config
        self.lower_limit = torch.tensor(config.lower_bound, dtype=torch.float32)
        self.max_limit = torch.tensor(config.upper_bound, dtype=torch.float32)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize a value x to the range [0, 1] based on bounds."""
        cx = torch.clamp(x, self.lower_limit, self.max_limit)
        return (cx - self.lower_limit) / (self.max_limit - self.lower_limit)
