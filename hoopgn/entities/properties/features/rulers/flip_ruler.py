from dataclasses import dataclass

import torch

from hoopgn.entities.properties.features.rulers.ruler import (
    PropertyRuler,
    PropertyRulerConfig,
)


@dataclass(kw_only=True)
class FlipRulerConfig(PropertyRulerConfig):
    pass


class FlipRuler(PropertyRuler):
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
