from dataclasses import dataclass
import math

import torch

from hoopgn.objects.properties.features.rulers.ruler import Ruler, RulerConfig


@dataclass(kw_only=True)
class AngularRulerConfig(RulerConfig):
    pass


class AngularRuler(Ruler):
    def distance(
        self,
        current: torch.Tensor,
        x: torch.Tensor,
    ) -> float:
        dot = torch.clamp(torch.abs(torch.dot(current, x)), -1.0, 1.0)
        return (2.0 * torch.arccos(dot) / math.pi).item()
