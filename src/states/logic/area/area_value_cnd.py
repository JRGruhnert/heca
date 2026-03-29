from dataclasses import dataclass

import torch

from src.states.logic.area.area import Area, AreaConfig
from src.states.logic.boundary import Boundary, BoundaryConfig
from src.states.logic.value_cnd import ValueCondition, ValueConditionConfig


@dataclass
class AreaValueConditionConfig(ValueConditionConfig):
    area: AreaConfig
    boundary: BoundaryConfig = BoundaryConfig(
        lower_bound=[-1.0, -1.0, -1.0], upper_bound=[1.0, 1.0, 1.0]
    )


class AreaValueCondition(ValueCondition):
    def __init__(
        self,
        config: AreaValueConditionConfig,
    ):
        self.boundary = Boundary(config.boundary)
        self.area = Area(config.area)

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """Check if the position is within the evaluation areas and normalize."""
        return self.boundary.normalize(x)

    def make_input(self, x: torch.Tensor) -> torch.Tensor:
        """Check if the position is within the evaluation areas and normalize."""
        nx = self.value(x)
        area_name = self.area.check_eval_area(x)
        one_hot_tensor = self.area.get_one_hot_area_vector(area_name)
        return torch.cat([nx, one_hot_tensor], dim=0)
