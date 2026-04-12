from dataclasses import dataclass

import torch

from hoopgn.entities.properties.states.area_state import AreaStateConfig, AreaState
from hoopgn.entities.properties.features.evaluators.evaluator import (
    PropertyEvaluator,
    PropertyEvaluatorConfig,
)


@dataclass(kw_only=True)
class AreaEvaluatorConfig(PropertyEvaluatorConfig):
    area: AreaStateConfig


class AreaEvaluator(PropertyEvaluator):
    def __init__(self, config: AreaEvaluatorConfig):
        self.config = config
        self.area = AreaState(config.area)

    def __call__(self, current: torch.Tensor, goal: torch.Tensor) -> bool:
        return self.area.check_area_similarity(current, goal)

    def is_in_area(self, value: torch.Tensor) -> bool:
        """Checks if the given value is within the defined areas."""
        return self.area.check_eval_area(value) is not None

    def validate(self, value: torch.Tensor) -> bool:
        """Checks if the given value is within the defined areas."""
        return self.area.check_eval_area(value) is not None
