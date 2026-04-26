from dataclasses import dataclass

import torch

from hoopgn.properties.rulers.ruler import (
    PropertyRuler,
)


class BinaryRuler(PropertyRuler):
    @dataclass(kw_only=True)
    class Config(PropertyRuler.Config):
        pass

    def distance(
        self,
        current: torch.Tensor,
        x: torch.Tensor,
    ) -> float:
        return torch.abs(current - x).item()
