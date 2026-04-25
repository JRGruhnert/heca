from dataclasses import dataclass

import torch

from hoopgn.entities.properties.rulers.ruler import (
    PropertyRuler,
)


class DomainRuler(PropertyRuler):
    @dataclass(kw_only=True)
    class Config(PropertyRuler.Config):
        pass

    def distance(self, x: torch.Tensor, y: torch.Tensor) -> float:
        raise NotImplementedError()
