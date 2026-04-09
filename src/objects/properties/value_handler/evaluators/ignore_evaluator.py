import torch
from dataclasses import dataclass

from src.objects.properties.value_handler.evaluators.evaluator import (
    StateEvaluator,
    StateEvaluatorConfig,
)


@dataclass
class IgnoreEvaluatorConfig(StateEvaluatorConfig):
    pass


class IgnoreEvaluator(StateEvaluator):
    def __init__(self, config: IgnoreEvaluatorConfig):
        self.config = config

    def __call__(self, current: torch.Tensor, goal: torch.Tensor) -> bool:
        return True
