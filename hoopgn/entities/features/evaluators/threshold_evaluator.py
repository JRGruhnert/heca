from dataclasses import dataclass

import torch

from hoopgn.properties.features.rulers.ruler import PropertyRulerConfig
from hoopgn.properties.features.evaluators.evaluator import (
    PropertyEvaluator,
    PropertyEvaluatorConfig,
)
from hoopgn.properties.features.rulers import select_property_ruler


@dataclass(kw_only=True)
class ThresholdEvaluatorConfig(PropertyEvaluatorConfig):
    ruler: PropertyRulerConfig
    threshold: float = 0.05


class ThresholdEvaluator(PropertyEvaluator):
    def __init__(self, config: ThresholdEvaluatorConfig):
        self.config = config
        self.ruler = select_property_ruler(config.ruler)

    def __call__(self, current: torch.Tensor, goal: torch.Tensor) -> bool:
        """Evaluate success condition based on Euclidean distance."""
        distance = self.ruler(current, goal)
        assert isinstance(distance, float), "Distance must be a float"
        return distance <= self.config.threshold
