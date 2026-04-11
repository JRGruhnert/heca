from dataclasses import dataclass

import torch

from hoopgn.objects.properties.area import Area, AreaConfig
from hoopgn.objects.properties.features.evaluators.evaluator import (
    PropertyEvaluator,
    PropertyEvaluatorConfig,
)


@dataclass(kw_only=True)
class AreaEvaluatorConfig(PropertyEvaluatorConfig):
    area: AreaConfig


class AreaEvaluator(PropertyEvaluator):
    def __init__(self, config: AreaEvaluatorConfig):
        self.config = config
        self.area = Area(config.area)

    def __call__(self, current: torch.Tensor, goal: torch.Tensor) -> bool:
        return self.area.check_area_similarity(current, goal)

    def is_in_area(self, value: torch.Tensor) -> bool:
        """Checks if the given value is within the defined areas."""
        return self.area.check_eval_area(value) is not None

    def validate(self, value: torch.Tensor) -> bool:
        """Checks if the given value is within the defined areas."""
        return self.area.check_eval_area(value) is not None
