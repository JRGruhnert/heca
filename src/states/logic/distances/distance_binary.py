from dataclasses import dataclass

import torch

from src.states.logic.distances.distance import Distance, ValueDistanceConfig


@dataclass
class ScalarDistanceConfig(ValueDistanceConfig):
    pass


class ScalarDistance(Distance):
    def distance(
        self,
        current: torch.Tensor,
        x: torch.Tensor,
    ) -> float:
        return torch.abs(current - x).item()
