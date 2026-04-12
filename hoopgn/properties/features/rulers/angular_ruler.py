from dataclasses import dataclass
import math

import torch

from hoopgn.properties.features.rulers.ruler import (
    PropertyRuler,
    PropertyRulerConfig,
)


@dataclass(kw_only=True)
class AngularRulerConfig(PropertyRulerConfig):
    pass


class AngularRuler(PropertyRuler):
    def distance(
        self,
        current: torch.Tensor,
        x: torch.Tensor,
    ) -> float:
        dot = torch.clamp(torch.abs(torch.dot(current, x)), -1.0, 1.0)
        return (2.0 * torch.arccos(dot) / math.pi).item()
