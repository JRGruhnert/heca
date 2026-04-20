from dataclasses import dataclass
import math
import torch

from hoopgn.properties.features.rulers.ruler import (
    PropertyRuler,
)


class EuclideanRuler(PropertyRuler):
    @dataclass(kw_only=True)
    class Config(PropertyRuler.Config):
        pass

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.max_dist = math.sqrt(3)

    def distance(
        self,
        current: torch.Tensor,
        x: torch.Tensor,
    ) -> float:
        return (torch.linalg.norm(current - x) / self.max_dist).item()
