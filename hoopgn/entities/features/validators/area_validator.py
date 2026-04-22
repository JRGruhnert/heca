from dataclasses import dataclass
import torch

from hoopgn.properties.states.area import (
    Area,
)
from hoopgn.properties.features.validators.validator import (
    PropertyValidator,
)


class AreaValidator(PropertyValidator):
    @dataclass(kw_only=True)
    class Config(PropertyValidator.Config):
        area: Area.Config

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.area = Area(cfg.area)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> bool:
        ax = self.area.label(x)
        ay = self.area.label(y)
        return ax is not None and ax == ay
