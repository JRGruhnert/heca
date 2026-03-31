from dataclasses import dataclass

import torch

from src.states.logic.area.area import Area, AreaConfig
from src.states.logic.eval_cnd import EvalCondition, EvaluationConfig


@dataclass
class AreaEvalConditionConfig(EvaluationConfig):
    area: AreaConfig


class AreaEvalCondition(EvalCondition):
    def __init__(self, config: AreaEvalConditionConfig):
        self.config = config
        self.area = Area(config.area)

    def evaluate(self, current: torch.Tensor, goal: torch.Tensor) -> bool:
        return self.area.check_area_similarity(current, goal)

    def is_in_area(self, value: torch.Tensor) -> bool:
        """Checks if the given value is within the defined areas."""
        return self.area.check_eval_area(value) is not None
