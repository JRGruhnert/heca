from dataclasses import dataclass

import torch

from src.factory import select_distance
from src.states.logic.distances.distance import ValueDistanceConfig
from src.states.logic.evaluations.evaluation import Evaluation, ValueEvaluationConfig


@dataclass
class ThresholdEvaluationConfig(ValueEvaluationConfig):
    distance: ValueDistanceConfig
    threshold: float = 0.05


class ThresholdEvaluation(Evaluation):
    def __init__(self, config: ThresholdEvaluationConfig):
        self.config = config
        self.condition = select_distance(config.distance)

    def __call__(self, current: torch.Tensor, goal: torch.Tensor) -> bool:
        """Evaluate success condition based on Euclidean distance."""
        distance = self.condition.distance(current, goal)
        assert isinstance(distance, float), "Distance must be a float"
        return distance <= self.config.threshold
