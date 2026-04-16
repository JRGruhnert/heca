from dataclasses import dataclass

import torch

from hoopgn.environments.properties.features.rulers.ruler import (
    PropertyRuler,
    PropertyRulerConfig,
)


@dataclass(kw_only=True)
class BinaryRulerConfig(PropertyRulerConfig):
    pass


class BinaryRuler(PropertyRuler):
    def distance(
        self,
        current: torch.Tensor,
        x: torch.Tensor,
    ) -> float:
        return torch.abs(current - x).item()
