import torch
from dataclasses import dataclass

from hoopgn.entities.properties.features.evaluators.evaluator import (
    PropertyEvaluator,
    PropertyEvaluatorConfig,
)


@dataclass(kw_only=True)
class IgnoreEvaluatorConfig(PropertyEvaluatorConfig):
    pass


class IgnoreEvaluator(PropertyEvaluator):
    def __init__(self, config: IgnoreEvaluatorConfig):
        self.config = config

    def __call__(self, current: torch.Tensor, goal: torch.Tensor) -> bool:
        return True
