from enum import Enum

from conf.calvin.states import CalvinStateSet

BASE = [
    CalvinStateSet.ee_position,
    CalvinStateSet.ee_rotation,
    CalvinStateSet.ee_scalar,
    CalvinStateSet.drawer_position,
    CalvinStateSet.drawer_rotation,
    CalvinStateSet.drawer_scalar,
    CalvinStateSet.button_position,
    CalvinStateSet.button_rotation,
    CalvinStateSet.button_scalar,
    CalvinStateSet.led_position,
    CalvinStateSet.led_rotation,
]

SLIDE = [
    CalvinStateSet.slide_position,
    CalvinStateSet.slide_rotation,
    CalvinStateSet.slide_scalar,
]

RED = [
    CalvinStateSet.block_red_position,
    CalvinStateSet.block_red_rotation,
    CalvinStateSet.block_red_scalar,
]

PINK = [
    CalvinStateSet.block_pink_position,
    CalvinStateSet.block_pink_rotation,
    CalvinStateSet.block_pink_scalar,
]

BLUE = [
    CalvinStateSet.block_blue_position,
    CalvinStateSet.block_blue_rotation,
    CalvinStateSet.block_blue_scalar,
]


class StateSet(Enum):
    BASE = "base"
    SLIDE = "slide"
    RED = "red"
    PINK = "pink"
    BLUE = "blue"
    SR = "sr"
    SRP = "srp"
    SRPB = "srpb"


STATES_SETS = {
    StateSet.BASE: BASE,
    StateSet.SLIDE: BASE + SLIDE,
    StateSet.RED: BASE + RED,
    StateSet.PINK: BASE + PINK,
    StateSet.BLUE: BASE + BLUE,
    StateSet.SR: BASE + SLIDE + RED,
    StateSet.SRP: BASE + SLIDE + RED + PINK,
    StateSet.SRPB: BASE + SLIDE + RED + PINK + BLUE,
}
