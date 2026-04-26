from abc import abstractmethod
from dataclasses import dataclass

import numpy as np
from hoopgn.misc.classes import SearchableClass
from hoopgn.misc.td import TDScene
from hoopgn.properties.property import Property
from hoopgn.converters.converter import HoopConverter, LeafConverter
from hoopgn.environments.calvins import calvin_properties


class Environment(SearchableClass):
    @dataclass(kw_only=True)
    class Config(SearchableClass.Config):
        hoop_cv: HoopConverter.Config
        leaf_cv: LeafConverter.Config

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.hoop_cv = HoopConverter.from_config(self.cfg.hoop_cv)
        self.leaf_cv = LeafConverter.from_config(self.cfg.leaf_cv)

    def to_observation(self, obs) -> TDScene:
        data = self.hoop_cv(obs)
        leaf = self.leaf_cv(obs)
        return TDScene(data=data, leaf=leaf)

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
    calvin_properties.ee_position,
    calvin_properties.ee_rotation,
    calvin_properties.ee_scalar,
    calvin_properties.drawer_position,
    calvin_properties.drawer_rotation,
    calvin_properties.drawer_scalar,
    calvin_properties.button_position,
    calvin_properties.button_rotation,
    calvin_properties.button_scalar,
    calvin_properties.led_position,
    calvin_properties.led_rotation,
]

_slide_base = [
    calvin_properties.slide_position,
    calvin_properties.slide_rotation,
    calvin_properties.slide_scalar,
]

_red_base = [
    calvin_properties.block_red_position,
    calvin_properties.block_red_rotation,
    calvin_properties.block_red_scalar,
]

_pink_base = [
    calvin_properties.block_pink_position,
    calvin_properties.block_pink_rotation,
    calvin_properties.block_pink_scalar,
]

_blue_base = [
    calvin_properties.block_blue_position,
    calvin_properties.block_blue_rotation,
    calvin_properties.block_blue_scalar,
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
