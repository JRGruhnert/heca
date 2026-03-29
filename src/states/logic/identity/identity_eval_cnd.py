import torch

from src.states.logic.eval_cnd import EvalCondition


class IgnoreEvalCondition(EvalCondition):
    """Success condition that always returns True (ignores the check)."""

    def evaluate(self, current: torch.Tensor, goal: torch.Tensor) -> bool:
        return True
