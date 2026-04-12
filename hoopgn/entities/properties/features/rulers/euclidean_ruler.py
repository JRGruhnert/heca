from dataclasses import dataclass
import math
import torch

from hoopgn.entities.properties.features.rulers.ruler import (
    PropertyRuler,
    PropertyRulerConfig,
)


@dataclass(kw_only=True)
class EuclideanRulerConfig(PropertyRulerConfig):
    pass


class EuclideanRuler(PropertyRuler):
    def __init__(self, config: EuclideanRulerConfig):
        super().__init__(config)
        self.config = config
        self.max_dist = math.sqrt(3)

    def distance(
        self,
        current: torch.Tensor,
        x: torch.Tensor,
    ) -> float:
        return (torch.linalg.norm(current - x) / self.max_dist).item()
