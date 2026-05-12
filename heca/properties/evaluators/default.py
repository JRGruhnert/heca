import torch
from dataclasses import dataclass

from heca.properties.evaluators.evaluator import (
    PropertyEvaluator,
)


class DefaultEvaluator(PropertyEvaluator):
    @dataclass(kw_only=True)
    class Config(PropertyEvaluator.Config):
        pass

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def __call__(self, x: torch.Tensor, y: torch.Tensor, distance: float) -> bool:
        return True
