from conf.agents.v1 import agents
from hoopgn.agents.agent import SkillConfig


_base = [
    agents.close_drawer,
    agents.close_drawer_back,
    agents.open_drawer,
    agents.open_drawer_back,
    agents.press_button,
    agents.press_button_back,
]

_slide_base = [
    agents.open_slide,
    agents.close_slide,
    agents.open_slide_back,
    agents.close_slide_back,
]

_red_base = [
    agents.pick_red_table,
    agents.place_red_table,
    agents.pick_red_drawer,
    agents.place_red_drawer,
]

_pink_base = [
    agents.pick_pink_table,
    agents.place_pink_table,
    agents.pick_pink_drawer,
    agents.place_pink_drawer,
]

_blue_base = [
    agents.pick_blue_table,
    agents.place_blue_table,
    agents.pick_blue_drawer,
    agents.place_blue_drawer,
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


def get_set(tag: str) -> list[SkillConfig]:
    assert tag in _sets, f"Unsupported skill set tag: {tag}"
    return _sets[tag]  # type: ignore
