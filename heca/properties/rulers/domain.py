from dataclasses import dataclass

import torch

from heca.properties.rulers.ruler import PropertyRuler


class DomainRuler(PropertyRuler):
    @dataclass(kw_only=True)
    class Config(PropertyRuler.Config):
        pass

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def distance(self, x: torch.Tensor, y: torch.Tensor) -> float:
        raise NotImplementedError()
