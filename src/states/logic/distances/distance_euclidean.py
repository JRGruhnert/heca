from dataclasses import dataclass
import math

import torch

from src.states.logic.distances.distance import Distance, ValueDistanceConfig


@dataclass
class EuclideanDistanceConfig(ValueDistanceConfig):
    pass


class EuclideanDistance(Distance):
    def __init__(self, config: EuclideanDistanceConfig):
        super().__init__(config)
        self.config = config
        self.max_dist = math.sqrt(3)

    def distance(
        self,
        current: torch.Tensor,
        x: torch.Tensor,
    ) -> float:
        return (torch.linalg.norm(current - x) / self.max_dist).item()
