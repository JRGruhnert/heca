from dataclasses import dataclass
import math

import torch

from src.states.logic.distance import Distance, DistanceConfig


@dataclass
class EuclideanDistanceConditionConfig(DistanceConfig):
    pass


class EuclideanDistanceCondition(Distance):
    def __init__(self, config: EuclideanDistanceConditionConfig):
        super().__init__(config)
        self.config = config
        self.max_dist = math.sqrt(3)

    def distance(
        self,
        current: torch.Tensor,
        x: torch.Tensor,
    ) -> float:
        return (torch.linalg.norm(current - x) / self.max_dist).item()
