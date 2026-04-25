from dataclasses import dataclass

import torch

from hoopgn.misc.area import Area
from hoopgn.entities.properties.evaluators.evaluator import (
    PropertyEvaluator,
)


class AreaEvaluator(PropertyEvaluator):
    @dataclass(kw_only=True)
    class Config(PropertyEvaluator.Config):
        area: Area.Config

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.area = Area.from_config(cfg.area)

    def __call__(self, x: torch.Tensor, y: torch.Tensor, distance: float) -> bool:
        return self.area.check_area_similarity(x, y)

    def is_in_area(self, value: torch.Tensor) -> bool:
        """Checks if the given value is within the defined areas."""
        return self.area.check_eval_area(value) is not None

    def validate(self, value: torch.Tensor) -> bool:
        """Checks if the given value is within the defined areas."""
        return self.area.check_eval_area(value) is not None
