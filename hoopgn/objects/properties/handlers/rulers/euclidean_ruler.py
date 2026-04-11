from dataclasses import dataclass
import math
import torch

from hoopgn.objects.properties.handlers.rulers.ruler import Ruler, RulerConfig


@dataclass(kw_only=True)
class EuclideanRulerConfig(RulerConfig):
    pass


class EuclideanRuler(Ruler):
    def __init__(self, config: EuclideanRulerConfig):
        super().__init__(config)
        self.config = config
        self.max_dist = math.sqrt(3)

    def distance(
        self,
        current: torch.Tensor,
        x: torch.Tensor,
    ) -> float:
        return (torch.linalg.norm(current - x) / self.max_dist).item()
