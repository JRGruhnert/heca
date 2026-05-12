from dataclasses import dataclass

import torch

from heca.properties.rulers.ruler import PropertyRuler


class FlipRuler(PropertyRuler):
    @dataclass(kw_only=True)
    class Config(PropertyRuler.Config):
        pass

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def distance(
        self,
        current: torch.Tensor,
        x: torch.Tensor,
    ) -> float:
        # TODO: Change that for hoopgn2.0
        return 0.0  # Always return zero distance for binary conditions
