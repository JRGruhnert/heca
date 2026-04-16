from conf.skills.v1 import skills
from hoopgn.agents.agent import SkillConfig


_base = [
    skills.close_drawer,
    skills.close_drawer_back,
    skills.open_drawer,
    skills.open_drawer_back,
    skills.press_button,
    skills.press_button_back,
]

_slide_base = [
    skills.open_slide,
    skills.close_slide,
    skills.open_slide_back,
    skills.close_slide_back,
]

_red_base = [
    skills.pick_red_table,
    skills.place_red_table,
    skills.pick_red_drawer,
    skills.place_red_drawer,
]

_pink_base = [
    skills.pick_pink_table,
    skills.place_pink_table,
    skills.pick_pink_drawer,
    skills.place_pink_drawer,
]

_blue_base = [
    skills.pick_blue_table,
    skills.place_blue_table,
    skills.pick_blue_drawer,
    skills.place_blue_drawer,
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
