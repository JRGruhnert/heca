from dataclasses import dataclass

import torch

from src.objects.properties.handlers.rulers.ruler import Ruler, RulerConfig


@dataclass
class FlipRulerConfig(RulerConfig):
    pass


class FlipRuler(Ruler):
    def __init__(self, config: FlipRulerConfig):
        super().__init__(config)
        self.config = config

    def distance(
        self,
        current: torch.Tensor,
        x: torch.Tensor,
    ) -> float:
        # TODO: Change that for hoopgn2.0
        return 0.0  # Always return zero distance for binary conditions
