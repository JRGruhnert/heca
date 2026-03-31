import torch

from src.states.logic.evaluations.evaluation import Evaluation


class IgnoreEvaluation(Evaluation):
    """Success condition that always returns True (ignores the check)."""

    def __call__(self, current: torch.Tensor, goal: torch.Tensor) -> bool:
        return True
