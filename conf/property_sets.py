from conf.hoopgnv1.properties.hoopgnv1 import PropertySet

BASE = [
    PropertySet.ee_position,
    PropertySet.ee_rotation,
    PropertySet.ee_scalar,
    PropertySet.drawer_position,
    PropertySet.drawer_rotation,
    PropertySet.drawer_scalar,
    PropertySet.button_position,
    PropertySet.button_rotation,
    PropertySet.button_scalar,
    PropertySet.led_position,
    PropertySet.led_rotation,
]

SLIDE = [
    PropertySet.slide_position,
    PropertySet.slide_rotation,
    PropertySet.slide_scalar,
]

RED = [
    PropertySet.block_red_position,
    PropertySet.block_red_rotation,
    PropertySet.block_red_scalar,
]

PINK = [
    PropertySet.block_pink_position,
    PropertySet.block_pink_rotation,
    PropertySet.block_pink_scalar,
]

BLUE = [
    PropertySet.block_blue_position,
    PropertySet.block_blue_rotation,
    PropertySet.block_blue_scalar,
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
