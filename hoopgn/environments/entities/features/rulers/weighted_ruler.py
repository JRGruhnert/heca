from dataclasses import dataclass

import torch

from hoopgn.environments.entities.features.rulers.ruler import (
    EntityRuler,
    EntityRulerConfig,
)


@dataclass(kw_only=True)
class IgnoreRulerConfig(EntityRulerConfig):
    default: float = 0.0


class IgnoreRuler(EntityRuler):
    def __init__(self, config: IgnoreRulerConfig):
        super().__init__(config)
        self.config = config

    def distance(
        self,
        current: torch.Tensor,
        x: torch.Tensor | None = None,
    ) -> float:
        return self.config.default
