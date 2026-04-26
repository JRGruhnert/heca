from dataclasses import dataclass, field

import torch

from hoopgn.misc.classes import ConfigurableClass
from hoopgn.properties.normalizers.boundary import (
    BoundaryNormalizer,
)
from hoopgn.misc.area import Area
from hoopgn.misc.state import State
from hoopgn.properties.default.v1.area import CalvinAreaConfig


class AreaGTModifier(ConfigurableClass):
    @dataclass(kw_only=True)
    class Config(ConfigurableClass.Config):
        area: Area.Config

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.state = State.from_area_config(cfg.area)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        one_hot = self.state(x)
        return torch.cat([x, one_hot], dim=0)


class AreaNormalizer(BoundaryNormalizer):
    @dataclass(kw_only=True)
    class Config(BoundaryNormalizer.Config):
        lower: list[float] = field(default_factory=lambda: [-1.0, -1.0, -1.0])
        upper: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
        modifier: AreaGTModifier.Config = AreaGTModifier.Config(
            area=CalvinAreaConfig(),
        )

    def __init__(
        self,
        cfg: Config,
    ):
        super().__init__(cfg)
        self.cfg = cfg
        self.lower = torch.tensor(cfg.lower, dtype=torch.float32)
        self.upper = torch.tensor(cfg.upper, dtype=torch.float32)
        self.modifier = AreaGTModifier.from_config(cfg.modifier)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        cx = torch.clamp(x, self.lower, self.upper)
        nx = (cx - self.lower) / (self.upper - self.lower)
        return self.modifier(nx)
