from hoopgn.agents.leafs.tapas import TapasAgent
from hoopgn.agents.leafs.reversed import RTapasAgent
from hoopgn.environments.calvin import CalvinEnvironment

close_drawer = TapasAgent.Config(
    parent=CalvinEnvironment.Query(),
    label="CloseDrawer",
    # id=0,
)
close_drawer_back = RTapasAgent.Query(
    parent=CalvinEnvironment.Query(),
    label="CloseDrawerBack",
    # id=1,
    overrides=[
        "ee_scalar",
        "ee_rotation",
        "ee_position",
    ],
)
open_drawer = TapasPolicy.Config(
    label="OpenDrawer",
    id=2,
)
open_drawer_back = RTapasAgent.Config(
    label="OpenDrawerBack",
    id=3,
    overrides=[
        "ee_scalar",
        "ee_rotation",
        "ee_position",
    ],
)
press_button = TapasPolicy.Config(
    label="PressButton",
    id=4,
)
press_button_back = RTapasAgent.Config(
    label="PressButtonBack",
    id=5,
    overrides=[
        "ee_scalar",
        "ee_rotation",
        "ee_position",
    ],
)
open_slide = TapasPolicy.Config(
    label="OpenSlide",
    id=6,
)
close_slide = TapasPolicy.Config(label="CloseSlide", id=7)
open_slide_back = RTapasAgent.Config(
    label="OpenSlideBack",
    id=8,
    overrides=[
        "ee_scalar",
        "ee_rotation",
        "ee_position",
    ],
)
close_slide_back = RTapasAgent.Config(
    label="CloseSlideBack",
    id=9,
    overrides=[
        "ee_scalar",
        "ee_rotation",
        "ee_position",
    ],
)
pick_red_table = TapasPolicy.Config(
    label="GrabRedTable",
    id=10,
)
place_red_table = RTapasAgent.Config(
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
pick_red_drawer = TapasPolicy.Config(
    label="GrabRedDrawer",
    id=12,
)
place_red_drawer = RTapasAgent.Config(
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
pick_pink_table = TapasPolicy.Config(
    label="GrabPinkTable",
    id=14,
)
place_pink_table = RTapasAgent.Config(
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
pick_pink_drawer = TapasPolicy.Config(
    label="GrabPinkDrawer",
    id=16,
)
place_pink_drawer = RTapasAgent.Config(
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
pick_blue_table = TapasPolicy.Config(
    label="GrabBlueTable",
    id=18,
)
pick_blue_drawer = TapasPolicy.Config(
    label="GrabBlueDrawer",
    id=19,
)
place_blue_table = RTapasAgent.Config(
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
place_blue_drawer = RTapasAgent.Config(
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
