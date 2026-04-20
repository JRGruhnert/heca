import torch
from dataclasses import dataclass

from hoopgn.environments.properties.features.evaluators.evaluator import (
    PropertyEvaluator,
)


class DefaultEvaluator(PropertyEvaluator):
    @dataclass(kw_only=True)
    class Config(PropertyEvaluator.Config):
        pass

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg

    def __call__(self, current: torch.Tensor, goal: torch.Tensor) -> bool:
        return True
