from dataclasses import dataclass

import torch

from heca.properties.rulers.ruler import PropertyRuler


class StateRuler(PropertyRuler):
    @dataclass(kw_only=True)
    class Config(PropertyRuler.Config):
        default_distance: float = 0.0

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg

    def distance(
        self,
        current: torch.Tensor,
        x: torch.Tensor | None = None,
    ) -> float:
        raise NotImplementedError()
