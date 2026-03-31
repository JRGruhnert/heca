from dataclasses import dataclass

import torch

from src.states.logic.area import Area, AreaConfig
from src.states.logic.boundary import Boundary, BoundaryConfig
from src.states.logic.values.value import Value, ValueConfig


@dataclass
class AreaValueConfig(ValueConfig):
    area: AreaConfig
    boundary: BoundaryConfig = BoundaryConfig(
        lower=[-1.0, -1.0, -1.0],
        upper=[1.0, 1.0, 1.0],
    )


class AreaValue(Value):
    def __init__(
        self,
        config: AreaValueConfig,
    ):
        self.boundary = Boundary(config.boundary)
        self.area = Area(config.area)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Check if the position is within the evaluation areas and normalize."""
        return self.boundary.normalize(x)

    def make_input(self, x: torch.Tensor) -> torch.Tensor:
        """Check if the position is within the evaluation areas and normalize."""
        nx = self.__call__(x)
        area_name = self.area.check_eval_area(x)
        one_hot_tensor = self.area.get_one_hot_area_vector(area_name)
        return torch.cat([nx, one_hot_tensor], dim=0)
