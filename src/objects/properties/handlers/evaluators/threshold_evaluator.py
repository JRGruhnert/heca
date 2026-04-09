from dataclasses import dataclass

import torch

from src.objects.properties.handlers.rulers.ruler import RulerConfig
from src.objects.properties.handlers.evaluators.evaluator import (
    StateEvaluator,
    StateEvaluatorConfig,
)
from src.objects.properties.handlers.rulers import select_state_ruler


@dataclass
class ThresholdEvaluatorConfig(StateEvaluatorConfig):
    ruler: RulerConfig
    threshold: float = 0.05


class ThresholdEvaluator(StateEvaluator):
    def __init__(self, config: ThresholdEvaluatorConfig):
        self.config = config
        self.ruler = select_state_ruler(config.ruler)

    def __call__(self, current: torch.Tensor, goal: torch.Tensor) -> bool:
        """Evaluate success condition based on Euclidean distance."""
        distance = self.ruler(current, goal)
        assert isinstance(distance, float), "Distance must be a float"
        return distance <= self.config.threshold
