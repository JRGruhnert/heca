from dataclasses import dataclass

import torch

from src.states.logic.distance import Distance, DistanceConfig


@dataclass
class FlipDistanceConditionConfig(DistanceConfig):
    pass


class SwitchDistanceCondition(Distance):
    def __init__(self, config: FlipDistanceConditionConfig):
        super().__init__(config)
        self.config = config

    def distance(
        self,
        current: torch.Tensor,
        x: torch.Tensor,
    ) -> float:
        return 0.0  # Always return zero distance for flip conditions
