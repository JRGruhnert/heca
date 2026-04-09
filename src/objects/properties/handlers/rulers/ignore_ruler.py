from dataclasses import dataclass

import torch

from src.objects.properties.handlers.rulers.ruler import Ruler, RulerConfig


@dataclass
class IgnoreRulerConfig(RulerConfig):
    default: float = 0.0


class IgnoreRuler(Ruler):
    def __init__(self, config: IgnoreRulerConfig):
        super().__init__(config)
        self.config = config

    def distance(
        self,
        current: torch.Tensor,
        x: torch.Tensor | None = None,
    ) -> float:
        return self.config.default

    def edge_feature(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Optional method to compute edge features for graph-based models."""
        return torch.tensor([self.distance(a, b), 1], dtype=torch.float32)
