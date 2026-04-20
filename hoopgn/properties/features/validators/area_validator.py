from dataclasses import dataclass
import torch

from hoopgn.properties.states.area_state import (
    AreaState,
)
from hoopgn.properties.features.validators.validator import (
    PropertyValidator,
)


class AreaValidator(PropertyValidator):
    @dataclass(kw_only=True)
    class Config(PropertyValidator.Config):
        area: AreaState.Config

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.area = AreaState(cfg.area)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> bool:
        ax = self.area.label(x)
        ay = self.area.label(y)
        return ax is not None and ax == ay
