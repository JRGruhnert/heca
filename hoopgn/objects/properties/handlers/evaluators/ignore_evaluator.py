import torch
from dataclasses import dataclass

from hoopgn.objects.properties.handlers.evaluators.evaluator import (
    StateEvaluator,
    StateEvaluatorConfig,
)


@dataclass(kw_only=True)
class IgnoreEvaluatorConfig(StateEvaluatorConfig):
    pass


class IgnoreEvaluator(StateEvaluator):
    def __init__(self, config: IgnoreEvaluatorConfig):
        self.config = config

    def __call__(self, current: torch.Tensor, goal: torch.Tensor) -> bool:
        return True
