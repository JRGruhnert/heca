from dataclasses import dataclass

from src.skills.tapas.tapas_leaf import TapasConfig


@dataclass
class TapasSkillSet:
    close_drawer = TapasConfig(
        label="CloseDrawer",
        id=0,
    )
    close_drawer_back = TapasConfig(
        label="CloseDrawerBack",
        id=1,
        overrides=[
            "ee_scalar",
            "ee_rotation",
            "ee_position",
        ],
    )
    open_drawer = TapasConfig(
        label="OpenDrawer",
        id=2,
    )
    open_drawer_back = TapasConfig(
        label="OpenDrawerBack",
        id=3,
        overrides=[
            "ee_scalar",
            "ee_rotation",
            "ee_position",
        ],
    )
    press_button = TapasConfig(
        label="PressButton",
        id=4,
    )
    press_button_back = TapasConfig(
        label="PressButtonBack",
        id=5,
        overrides=[
            "ee_scalar",
            "ee_rotation",
            "ee_position",
        ],
    )
    open_slide = TapasConfig(
        label="OpenSlide",
        id=6,
    )
    close_slide = TapasConfig(label="CloseSlide", id=7)
    open_slide_back = TapasConfig(
        label="OpenSlideBack",
        id=8,
        overrides=[
            "ee_scalar",
            "ee_rotation",
            "ee_position",
        ],
    )
    close_slide_back = TapasConfig(
        label="CloseSlideBack",
        id=9,
        overrides=[
            "ee_scalar",
            "ee_rotation",
            "ee_position",
        ],
    )
    pick_red_table = TapasConfig(
        label="GrabRedTable",
        id=10,
    )
    place_red_table = TapasConfig(
        label="PlaceRedTable",
        id=11,
        overrides=[
            "ee_scalar",
            "ee_rotation",
            "ee_position",
            "block_red_position",
            "block_red_rotation",
            "block_red_scalar",
        ],
    )
    pick_red_drawer = TapasConfig(
        label="GrabRedDrawer",
        id=12,
    )
    place_red_drawer = TapasConfig(
        label="PlaceRedDrawer",
        id=13,
        overrides=[
            "ee_scalar",
            "ee_rotation",
            "ee_position",
            "block_red_position",
            "block_red_rotation",
            "block_red_scalar",
        ],
    )
    pick_pink_table = TapasConfig(
        label="GrabPinkTable",
        id=14,
    )
    place_pink_table = TapasConfig(
        label="PlacePinkTable",
        id=15,
        overrides=[
            "ee_scalar",
            "ee_rotation",
            "ee_position",
            "block_pink_position",
            "block_pink_rotation",
            "block_pink_scalar",
        ],
    )
    pick_pink_drawer = TapasConfig(
        label="GrabPinkDrawer",
        id=16,
    )
    place_pink_drawer = TapasConfig(
        label="PlacePinkDrawer",
        id=17,
        overrides=[
            "ee_scalar",
            "ee_rotation",
            "ee_position",
            "block_pink_position",
            "block_pink_rotation",
            "block_pink_scalar",
        ],
    )
    pick_blue_table = TapasConfig(
        label="GrabBlueTable",
        id=18,
    )
    pick_blue_drawer = TapasConfig(
        label="GrabBlueDrawer",
        id=19,
    )
    place_blue_table = TapasConfig(
        label="PlaceBlueTable",
        id=20,
        overrides=[
            "ee_scalar",
            "ee_rotation",
            "ee_position",
            "block_blue_position",
            "block_blue_rotation",
            "block_blue_scalar",
        ],
    )
    place_blue_drawer = TapasConfig(
        label="PlaceBlueDrawer",
        id=21,
        overrides=[
            "ee_scalar",
            "ee_rotation",
            "ee_position",
            "block_blue_position",
            "block_blue_rotation",
            "block_blue_scalar",
        ],
    )
