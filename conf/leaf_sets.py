from enum import Enum

from conf.custom.skills.tapas_skills import TapaLeafSet


BASE = [
    TapaLeafSet.close_drawer,
    TapaLeafSet.close_drawer_back,
    TapaLeafSet.open_drawer,
    TapaLeafSet.open_drawer_back,
    TapaLeafSet.press_button,
    TapaLeafSet.press_button_back,
]

SLIDE = [
    TapaLeafSet.open_slide,
    TapaLeafSet.close_slide,
    TapaLeafSet.open_slide_back,
    TapaLeafSet.close_slide_back,
]

RED = [
    TapaLeafSet.pick_red_table,
    TapaLeafSet.place_red_table,
    TapaLeafSet.pick_red_drawer,
    TapaLeafSet.place_red_drawer,
]

PINK = [
    TapaLeafSet.pick_pink_table,
    TapaLeafSet.place_pink_table,
    TapaLeafSet.pick_pink_drawer,
    TapaLeafSet.place_pink_drawer,
]

BLUE = [
    TapaLeafSet.pick_blue_table,
    TapaLeafSet.place_blue_table,
    TapaLeafSet.pick_blue_drawer,
    TapaLeafSet.place_blue_drawer,
]


class LeafSet(Enum):
    BASE = "base"
    SLIDE = "slide"
    RED = "red"
    PINK = "pink"
    BLUE = "blue"
    SR = "sr"
    SRP = "srp"
    SRPB = "srpb"


SKILL_SETS = {
    LeafSet.BASE: BASE,
    LeafSet.SLIDE: BASE + SLIDE,
    LeafSet.RED: BASE + RED,
    LeafSet.PINK: BASE + PINK,
    LeafSet.BLUE: BASE + BLUE,
    LeafSet.SR: BASE + SLIDE + RED,
    LeafSet.SRP: BASE + SLIDE + RED + PINK,
    LeafSet.SRPB: BASE + SLIDE + RED + PINK + BLUE,
}
# if storage.config.skills == SET_SLIDE:
#          num_skills = 6
#      elif (
#          storage.config.skills == SET_RED
#           or storage.config.skills == SET_BLUE
#           or storage.config.skills == SET_PINK
#        ):
#            num_skills = 8
#        elif storage.config.skills == SET_SR:
#            num_skills = 10
#        elif storage.config.skills == SET_SRP:
#            num_skills = 12
#        elif storage.config.skills == SET_SRPB:
#            num_skills = 14
