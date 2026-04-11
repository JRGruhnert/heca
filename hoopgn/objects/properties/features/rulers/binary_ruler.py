from dataclasses import dataclass

import torch

from hoopgn.objects.properties.features.rulers.ruler import Ruler, RulerConfig


@dataclass(kw_only=True)
class BinaryRulerConfig(RulerConfig):
    pass


class BinaryRuler(Ruler):
    def distance(
        self,
        current: torch.Tensor,
        x: torch.Tensor,
    ) -> float:
        return torch.abs(current - x).item()
