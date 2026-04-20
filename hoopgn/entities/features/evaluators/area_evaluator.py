from dataclasses import dataclass

import torch

from hoopgn.properties.states.area_state import AreaState
from hoopgn.properties.features.evaluators.evaluator import (
    PropertyEvaluator,
)


class AreaEvaluator(PropertyEvaluator):
    @dataclass(kw_only=True)
    class Config(PropertyEvaluator.Config):
        area: AreaState.Config

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.area = AreaState(cfg.area)

    def __call__(self, current: torch.Tensor, goal: torch.Tensor) -> bool:
        return self.area.check_area_similarity(current, goal)

    def is_in_area(self, value: torch.Tensor) -> bool:
        """Checks if the given value is within the defined areas."""
        return self.area.check_eval_area(value) is not None

    def validate(self, value: torch.Tensor) -> bool:
        """Checks if the given value is within the defined areas."""
        return self.area.check_eval_area(value) is not None
