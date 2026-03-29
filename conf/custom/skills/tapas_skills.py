from dataclasses import dataclass

from src.skills.tree.leafs.leaf_tapas import TapasLeafConfig


@dataclass
class TapaLeafSet:
    close_drawer = TapasLeafConfig(
        label="CloseDrawer",
        id=0,
        reversed=False,
        predict_as_batch=True,
        overrides=[],
    )
    close_drawer_back = TapasLeafConfig(
        label="CloseDrawerBack",
        id=1,
        reversed=True,
        predict_as_batch=True,
        overrides=["ee_scalar", "ee_rotation", "ee_position"],
    )
    open_drawer = TapasLeafConfig(
        label="OpenDrawer",
        id=2,
        reversed=False,
        predict_as_batch=True,
        overrides=[],
    )
    open_drawer_back = TapasLeafConfig(
        label="OpenDrawerBack",
        id=3,
        reversed=True,
        predict_as_batch=True,
        overrides=["ee_scalar", "ee_rotation", "ee_position"],
    )
    press_button = TapasLeafConfig(
        label="PressButton",
        id=4,
        reversed=False,
        predict_as_batch=True,
        overrides=[],
    )
    press_button_back = TapasLeafConfig(
        label="PressButtonBack",
        id=5,
        reversed=True,
        predict_as_batch=True,
        overrides=["ee_scalar", "ee_rotation", "ee_position"],
    )
    open_slide = TapasLeafConfig(
        label="OpenSlide",
        id=6,
        reversed=False,
        predict_as_batch=True,
        overrides=[],
    )
    close_slide = TapasLeafConfig(
        label="CloseSlide",
        id=7,
        reversed=False,
        predict_as_batch=True,
        overrides=[],
    )
    open_slide_back = TapasLeafConfig(
        label="OpenSlideBack",
        id=8,
        reversed=True,
        predict_as_batch=True,
        overrides=["ee_scalar", "ee_rotation", "ee_position"],
    )
    close_slide_back = TapasLeafConfig(
        label="CloseSlideBack",
        id=9,
        reversed=True,
        predict_as_batch=True,
        overrides=["ee_scalar", "ee_rotation", "ee_position"],
    )
    pick_red_table = TapasLeafConfig(
        label="GrabRedTable",
        id=10,
        reversed=False,
        predict_as_batch=True,
        overrides=[],
    )
    place_red_table = TapasLeafConfig(
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
    pick_red_drawer = TapasLeafConfig(
        label="GrabRedDrawer",
        id=12,
        reversed=False,
        predict_as_batch=True,
        overrides=[],
    )
    place_red_drawer = TapasLeafConfig(
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
    pick_pink_table = TapasLeafConfig(
        label="GrabPinkTable",
        id=14,
        reversed=False,
        predict_as_batch=True,
        overrides=[],
    )
    place_pink_table = TapasLeafConfig(
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
    pick_pink_drawer = TapasLeafConfig(
        label="GrabPinkDrawer",
        id=16,
        reversed=False,
        predict_as_batch=True,
        overrides=[],
    )
    place_pink_drawer = TapasLeafConfig(
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
    pick_blue_table = TapasLeafConfig(
        label="GrabBlueTable",
        id=18,
        reversed=False,
        predict_as_batch=True,
        overrides=[],
    )
    pick_blue_drawer = TapasLeafConfig(
        label="GrabBlueDrawer",
        id=19,
        reversed=False,
        predict_as_batch=True,
        overrides=[],
    )
    place_blue_table = TapasLeafConfig(
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
    place_blue_drawer = TapasLeafConfig(
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
