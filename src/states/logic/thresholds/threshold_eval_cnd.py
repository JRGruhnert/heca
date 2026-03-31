from dataclasses import dataclass

import torch

from src.factory import select_distance
from src.states.logic.distance import DistanceConfig
from src.states.logic.eval_cnd import EvalCondition, EvaluationConfig


@dataclass
class ThresholdEvalConditionConfig(EvaluationConfig):
    distance: DistanceConfig
    threshold: float = 0.05


class ThresholdEvalCondition(EvalCondition):
    def __init__(self, config: ThresholdEvalConditionConfig):
        self.config = config
        self.condition = select_distance(config.distance)

    def evaluate(self, current: torch.Tensor, goal: torch.Tensor) -> bool:
        """Evaluate success condition based on Euclidean distance."""
        distance = self.condition.distance(current, goal)
        assert isinstance(distance, float), "Distance must be a float"
        return distance <= self.config.threshold
