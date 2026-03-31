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


class ObjectSet(Enum):
    BASE = "base"
    SLIDE = "slide"
    RED = "red"
    PINK = "pink"
    BLUE = "blue"
    SR = "sr"
    SRP = "srp"
    SRPB = "srpb"


OBJECT_SETS = {
    ObjectSet.BASE: BASE,
    ObjectSet.SLIDE: BASE + SLIDE,
    ObjectSet.RED: BASE + RED,
    ObjectSet.PINK: BASE + PINK,
    ObjectSet.BLUE: BASE + BLUE,
    ObjectSet.SR: BASE + SLIDE + RED,
    ObjectSet.SRP: BASE + SLIDE + RED + PINK,
    ObjectSet.SRPB: BASE + SLIDE + RED + PINK + BLUE,
}
