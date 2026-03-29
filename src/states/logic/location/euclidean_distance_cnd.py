from dataclasses import dataclass
import math

import torch

from src.states.logic.distance_cnd import DistanceCondition, DistanceConditionConfig


@dataclass
class EuclideanDistanceConditionConfig(DistanceConditionConfig):
    pass


class EuclideanDistanceCondition(DistanceCondition):
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
