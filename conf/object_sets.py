from enum import Enum

from conf.hoopgnv1.states.hoopgnv1 import CalvinStateSet

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

OBJECT_SETS = {
    "base": BASE,
    "slider": BASE + SLIDE,
    "red": BASE + RED,
    "pink": BASE + PINK,
    "blue": BASE + BLUE,
    "sr": BASE + SLIDE + RED,
    "srp": BASE + SLIDE + RED + PINK,
    "srpb": BASE + SLIDE + RED + PINK + BLUE,
}
