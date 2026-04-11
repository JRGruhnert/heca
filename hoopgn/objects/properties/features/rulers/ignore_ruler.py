from dataclasses import dataclass

import torch

from hoopgn.objects.properties.features.rulers.ruler import (
    PropertyRuler,
    PropertyRulerConfig,
)


@dataclass(kw_only=True)
class IgnoreRulerConfig(PropertyRulerConfig):
    default: float = 0.0


class IgnoreRuler(PropertyRuler):
    def __init__(self, config: IgnoreRulerConfig):
        super().__init__(config)
        self.config = config

    def distance(
        self,
        current: torch.Tensor,
        x: torch.Tensor | None = None,
    ) -> float:
        return self.config.default
