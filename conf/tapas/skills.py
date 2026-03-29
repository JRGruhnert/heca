from dataclasses import dataclass
from enum import Enum

from src.skills.tapas import TapasSkillConfig


@dataclass
class TapasSkillSet:
    close_drawer = TapasSkillConfig(
        label="CloseDrawer",
        id=0,
        reversed=False,
        predict_as_batch=True,
        overrides=[],
    )
    close_drawer_back = TapasSkillConfig(
        label="CloseDrawerBack",
        id=1,
        reversed=True,
        predict_as_batch=True,
        overrides=["ee_scalar", "ee_rotation", "ee_position"],
    )
    open_drawer = TapasSkillConfig(
        label="OpenDrawer",
        id=2,
        reversed=False,
        predict_as_batch=True,
        overrides=[],
    )
    open_drawer_back = TapasSkillConfig(
        label="OpenDrawerBack",
        id=3,
        reversed=True,
        predict_as_batch=True,
        overrides=["ee_scalar", "ee_rotation", "ee_position"],
    )
    press_button = TapasSkillConfig(
        label="PressButton",
        id=4,
        reversed=False,
        predict_as_batch=True,
        overrides=[],
    )
    press_button_back = TapasSkillConfig(
        label="PressButtonBack",
        id=5,
        reversed=True,
        predict_as_batch=True,
        overrides=["ee_scalar", "ee_rotation", "ee_position"],
    )
    open_slide = TapasSkillConfig(
        label="OpenSlide",
        id=6,
        reversed=False,
        predict_as_batch=True,
        overrides=[],
    )
    close_slide = TapasSkillConfig(
        label="CloseSlide",
        id=7,
        reversed=False,
        predict_as_batch=True,
        overrides=[],
    )
    open_slide_back = TapasSkillConfig(
        label="OpenSlideBack",
        id=8,
        reversed=True,
        predict_as_batch=True,
        overrides=["ee_scalar", "ee_rotation", "ee_position"],
    )
    close_slide_back = TapasSkillConfig(
        label="CloseSlideBack",
        id=9,
        reversed=True,
        predict_as_batch=True,
        overrides=["ee_scalar", "ee_rotation", "ee_position"],
    )
    pick_red_table = TapasSkillConfig(
        label="GrabRedTable",
        id=10,
        reversed=False,
        predict_as_batch=True,
        overrides=[],
    )
    place_red_table = TapasSkillConfig(
        label="PlaceRedTable",
        id=11,
        reversed=True,
        predict_as_batch=True,
        overrides=[
            "ee_scalar",
            "ee_rotation",
            "ee_position",
            "block_red_position",
            "block_red_rotation",
            "block_red_scalar",
        ],
    )
    pick_red_drawer = TapasSkillConfig(
        label="GrabRedDrawer",
        id=12,
        reversed=False,
        predict_as_batch=True,
        overrides=[],
    )
    place_red_drawer = TapasSkillConfig(
        label="PlaceRedDrawer",
        id=13,
        reversed=True,
        predict_as_batch=True,
        overrides=[
            "ee_scalar",
            "ee_rotation",
            "ee_position",
            "block_red_position",
            "block_red_rotation",
            "block_red_scalar",
        ],
    )
    pick_pink_table = TapasSkillConfig(
        label="GrabPinkTable",
        id=14,
        reversed=False,
        predict_as_batch=True,
        overrides=[],
    )
    place_pink_table = TapasSkillConfig(
        label="PlacePinkTable",
        id=15,
        reversed=True,
        predict_as_batch=True,
        overrides=[
            "ee_scalar",
            "ee_rotation",
            "ee_position",
            "block_pink_position",
            "block_pink_rotation",
            "block_pink_scalar",
        ],
    )
    pick_pink_drawer = TapasSkillConfig(
        label="GrabPinkDrawer",
        id=16,
        reversed=False,
        predict_as_batch=True,
        overrides=[],
    )
    place_pink_drawer = TapasSkillConfig(
        label="PlacePinkDrawer",
        id=17,
        reversed=True,
        predict_as_batch=True,
        overrides=[
            "ee_scalar",
            "ee_rotation",
            "ee_position",
            "block_pink_position",
            "block_pink_rotation",
            "block_pink_scalar",
        ],
    )
    pick_blue_table = TapasSkillConfig(
        label="GrabBlueTable",
        id=18,
        reversed=False,
        predict_as_batch=True,
        overrides=[],
    )
    pick_blue_drawer = TapasSkillConfig(
        label="GrabBlueDrawer",
        id=19,
        reversed=False,
        predict_as_batch=True,
        overrides=[],
    )
    place_blue_table = TapasSkillConfig(
        label="PlaceBlueTable",
        id=20,
        reversed=True,
        predict_as_batch=True,
        overrides=[
            "ee_scalar",
            "ee_rotation",
            "ee_position",
            "block_blue_position",
            "block_blue_rotation",
            "block_blue_scalar",
        ],
    )
    place_blue_drawer = TapasSkillConfig(
        label="PlaceBlueDrawer",
        id=21,
        reversed=True,
        predict_as_batch=True,
        overrides=[
            "ee_scalar",
            "ee_rotation",
            "ee_position",
            "block_blue_position",
            "block_blue_rotation",
            "block_blue_scalar",
        ],
    )
