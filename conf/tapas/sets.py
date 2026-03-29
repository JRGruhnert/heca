from enum import Enum

from conf.tapas.skills import TapasSkillSet


BASE = [
    TapasSkillSet.close_drawer,
    TapasSkillSet.close_drawer_back,
    TapasSkillSet.open_drawer,
    TapasSkillSet.open_drawer_back,
    TapasSkillSet.press_button,
    TapasSkillSet.press_button_back,
]

SLIDE = [
    TapasSkillSet.open_slide,
    TapasSkillSet.close_slide,
    TapasSkillSet.open_slide_back,
    TapasSkillSet.close_slide_back,
]

RED = [
    TapasSkillSet.pick_red_table,
    TapasSkillSet.place_red_table,
    TapasSkillSet.pick_red_drawer,
    TapasSkillSet.place_red_drawer,
]

PINK = [
    TapasSkillSet.pick_pink_table,
    TapasSkillSet.place_pink_table,
    TapasSkillSet.pick_pink_drawer,
    TapasSkillSet.place_pink_drawer,
]

BLUE = [
    TapasSkillSet.pick_blue_table,
    TapasSkillSet.place_blue_table,
    TapasSkillSet.pick_blue_drawer,
    TapasSkillSet.place_blue_drawer,
]


class SkillSet(Enum):
    BASE = "base"
    SLIDE = "slide"
    RED = "red"
    PINK = "pink"
    BLUE = "blue"
    SR = "sr"
    SRP = "srp"
    SRPB = "srpb"


SKILL_SETS = {
    SkillSet.BASE: BASE,
    SkillSet.SLIDE: BASE + SLIDE,
    SkillSet.RED: BASE + RED,
    SkillSet.PINK: BASE + PINK,
    SkillSet.BLUE: BASE + BLUE,
    SkillSet.SR: BASE + SLIDE + RED,
    SkillSet.SRP: BASE + SLIDE + RED + PINK,
    SkillSet.SRPB: BASE + SLIDE + RED + PINK + BLUE,
}
