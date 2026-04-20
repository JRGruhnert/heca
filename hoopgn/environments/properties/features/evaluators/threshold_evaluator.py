from dataclasses import dataclass

import torch

from hoopgn.environments.properties.features.evaluators.evaluator import (
    PropertyEvaluator,
)
from hoopgn.environments.properties.features.rulers.ruler import PropertyRuler


class ThresholdEvaluator(PropertyEvaluator):
    @dataclass(kw_only=True)
    class Config(PropertyEvaluator.Config):
        ruler: PropertyRuler.Config
        threshold: float = 0.05

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.ruler = PropertyRuler.from_config(cfg.ruler)

    def __call__(self, current: torch.Tensor, goal: torch.Tensor) -> bool:
        """Evaluate success condition based on Euclidean distance."""
        distance = self.ruler(current, goal)
        assert isinstance(distance, float), "Distance must be a float"
        return distance <= self.cfg.threshold
