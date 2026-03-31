from dataclasses import dataclass
import math

import torch

from src.states.logic.distances.distance import Distance, DistanceConfig


@dataclass
class AngularDistanceConfig(DistanceConfig):
    pass


class AngularDistance(Distance):
    def distance(
        self,
        current: torch.Tensor,
        x: torch.Tensor,
    ) -> float:
        dot = torch.clamp(torch.abs(torch.dot(current, x)), -1.0, 1.0)
        return (2.0 * torch.arccos(dot) / math.pi).item()
