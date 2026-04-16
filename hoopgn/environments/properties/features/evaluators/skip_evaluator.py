import torch
from dataclasses import dataclass

from hoopgn.environments.properties.features.evaluators.evaluator import (
    PropertyEvaluator,
    PropertyEvaluatorConfig,
)


@dataclass(kw_only=True)
class PIgnoreEvaluatorConfig(PropertyEvaluatorConfig):
    pass


class PIgnoreEvaluator(PropertyEvaluator):
    def __init__(self, config: PIgnoreEvaluatorConfig):
        self.config = config

    def __call__(self, current: torch.Tensor, goal: torch.Tensor) -> bool:
        return True
