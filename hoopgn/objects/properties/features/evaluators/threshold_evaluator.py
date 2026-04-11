from dataclasses import dataclass

import torch

from hoopgn.objects.properties.features.rulers.ruler import RulerConfig
from hoopgn.objects.properties.features.evaluators.evaluator import (
    StateEvaluator,
    StateEvaluatorConfig,
)
from hoopgn.objects.properties.features.rulers import select_property_ruler


@dataclass(kw_only=True)
class ThresholdEvaluatorConfig(StateEvaluatorConfig):
    ruler: RulerConfig
    threshold: float = 0.05


class ThresholdEvaluator(StateEvaluator):
    def __init__(self, config: ThresholdEvaluatorConfig):
        self.config = config
        self.ruler = select_property_ruler(config.ruler)

    def __call__(self, current: torch.Tensor, goal: torch.Tensor) -> bool:
        """Evaluate success condition based on Euclidean distance."""
        distance = self.ruler(current, goal)
        assert isinstance(distance, float), "Distance must be a float"
        return distance <= self.config.threshold
