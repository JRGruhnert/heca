from dataclasses import dataclass

import torch

from src.states.logic.distance_cnd import DistanceCondition, DistanceConditionConfig


@dataclass
class RangeDistanceConditionConfig(DistanceConditionConfig):
    pass


class RangeDistanceCondition(DistanceCondition):
    def distance(
        self,
        current: torch.Tensor,
        x: torch.Tensor,
    ) -> float:
        return torch.abs(current - x).item()
