from dataclasses import dataclass

import torch

from src.factory import select_distance_condition
from src.states.logic.distance_cnd import DistanceConditionConfig
from src.states.logic.eval_cnd import EvalCondition, EvalConditionConfig


@dataclass
class PreciseEvalConditionConfig(EvalConditionConfig):
    distance: DistanceConditionConfig
    threshold: float = 0.05


class PreciseEvalCondition(EvalCondition):
    def __init__(self, config: PreciseEvalConditionConfig):
        self.config = config
        self.condition = select_distance_condition(config.distance)

    def evaluate(self, current: torch.Tensor, goal: torch.Tensor) -> bool:
        """Evaluate success condition based on Euclidean distance."""
        distance = self.condition.distance(current, goal)
        assert isinstance(distance, float), "Distance must be a float"
        return distance <= self.config.threshold
