from dataclasses import dataclass

import torch

from src.states.logic.distances.distance import Distance, ValueDistanceConfig


@dataclass
class FlipDistanceConfig(ValueDistanceConfig):
    pass


class FlipDistance(Distance):
    def __init__(self, config: FlipDistanceConfig):
        super().__init__(config)
        self.config = config

    def distance(
        self,
        current: torch.Tensor,
        x: torch.Tensor,
    ) -> float:
        # TODO: Change that for hoopgn2.0
        return 0.0  # Always return zero distance for binary conditions
