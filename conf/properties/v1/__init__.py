from conf.properties.v1 import properties
from hoopgn.environments.properties.property import PropertyConfig

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


def get_set(tag: str) -> list[PropertyConfig]:
    assert tag in _sets, f"Unsupported property set tag: {tag}"
    return _sets[tag]  # type: ignore
