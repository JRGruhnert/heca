from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from hoopgn.base import RegisterableClass


class Environment(RegisterableClass):
    @dataclass(kw_only=True)
    class Config(RegisterableClass.Config):
        pass

    def __init__(self, cfg: Config):
        self.cfg = cfg

    @abstractmethod
    def sample(self) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def step(self, action: np.ndarray) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def render(self):
        raise NotImplementedError()

    @abstractmethod
    def close(self):
        raise NotImplementedError()


from hoopgn.domains.calvins import calvin_properties
from hoopgn.properties.property import Property

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
