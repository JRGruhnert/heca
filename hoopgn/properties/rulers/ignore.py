from dataclasses import dataclass

import torch

from hoopgn.properties.rulers.ruler import (
    PropertyRuler,
)


class IgnoreRuler(PropertyRuler):
    @dataclass(kw_only=True)
    class Config(PropertyRuler.Config):
        default_distance: float = 0.0

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg

    def distance(
        self,
        current: torch.Tensor,
        x: torch.Tensor | None = None,
    ) -> float:
        return self.cfg.default_distance
