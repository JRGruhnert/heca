from abc import abstractmethod
from dataclasses import dataclass

import numpy as np
from hoopgn.classes import SearchableClass
from hoopgn.misc.td import TDScene
from hoopgn.properties.property import Property
from hoopgn.converters.tapas import TapasConverter
from hoopgn.converters.v1 import V1Converter
from hoopgn.converters.v2 import V2Converter
from hoopgn.domains.calvins import properties


class Environment(SearchableClass):
    @dataclass(kw_only=True)
    class Config(SearchableClass.Config):
        v1_cv: V1Converter.Config
        v2_cv: V2Converter.Config
        ts_cv: TapasConverter.Config

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.v1_cv = V1Converter.from_config(self.cfg.v1_cv)
        self.v2_cv = V2Converter.from_config(self.cfg.v2_cv)
        self.ts_cv = TapasConverter.from_config(self.cfg.ts_cv)

    def to_observation(self, obs) -> TDScene:
        return TDScene(
            {
                self.v1_cv.cfg.label: self.v1_cv(obs),
                self.v2_cv.cfg.label: self.v2_cv(obs),
                self.ts_cv.cfg.label: self.ts_cv(obs),
            }
        )

    @abstractmethod
    def sample(self) -> TDScene:
        raise NotImplementedError()

    @abstractmethod
    def step(self, action: np.ndarray) -> TDScene:
        raise NotImplementedError()

    @abstractmethod
    def render(self):
        raise NotImplementedError()

    @abstractmethod
    def close(self):
        raise NotImplementedError()


_base = [
    properties.ee_position,
    properties.ee_rotation,
    properties.ee_scalar,
    properties.drawer_position,
    properties.drawer_rotation,
    properties.drawer_scalar,
    properties.button_position,
    properties.button_rotation,
    properties.button_scalar,
    properties.led_position,
    properties.led_rotation,
]

_slide_base = [
    properties.slide_position,
    properties.slide_rotation,
    properties.slide_scalar,
]

_red_base = [
    properties.block_red_position,
    properties.block_red_rotation,
    properties.block_red_scalar,
]

_pink_base = [
    properties.block_pink_position,
    properties.block_pink_rotation,
    properties.block_pink_scalar,
]

_blue_base = [
    properties.block_blue_position,
    properties.block_blue_rotation,
    properties.block_blue_scalar,
]

_sets = {
    "base": _base,
    "slider": _base + _slide_base,
    "red": _base + _red_base,
    "pink": _base + _pink_base,
    "blue": _base + _blue_base,
    "sr": _base + _slide_base + _red_base,
    "srp": _base + _slide_base + _red_base + _pink_base,
    "srpb": _base + _slide_base + _red_base + _pink_base + _blue_base,
}


def get_set(tag: str) -> list[Property.Config]:
    assert tag in _sets, f"Unsupported property set tag: {tag}"
    return _sets[tag]  # type: ignore
