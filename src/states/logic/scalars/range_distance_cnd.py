from dataclasses import dataclass

import torch

from src.states.logic.distance import Distance, DistanceConfig


@dataclass
class RangeDistanceConditionConfig(DistanceConfig):
    pass


class RangeDistanceCondition(Distance):
    def distance(
        self,
        current: torch.Tensor,
        x: torch.Tensor,
    ) -> float:
        return torch.abs(current - x).item()
