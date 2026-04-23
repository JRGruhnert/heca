from dataclasses import dataclass

import torch

from hoopgn.properties.evaluators.evaluator import (
    PropertyEvaluator,
)


class ThresholdEvaluator(PropertyEvaluator):
    @dataclass(kw_only=True)
    class Config(PropertyEvaluator.Config):
        threshold: float = 0.05

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg

    def __call__(self, x: torch.Tensor, y: torch.Tensor, distance: float) -> bool:
        return distance <= self.cfg.threshold
