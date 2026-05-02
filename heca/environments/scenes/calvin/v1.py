from heca.properties.default.v1.area import AreaProperty
from heca.properties.default.v1.bool import BoolProperty
from heca.properties.default.v1.flip import FlipProperty
from heca.properties.default.v1.position import PositionProperty
from heca.properties.default.v1.range import RangeProperty
from heca.properties.default.v1.rotation import RotationProperty
from heca.properties.default.v1.rotation_ignore import RotationIgnoreProperty

v1_pos_ee = PositionProperty.Config(label="ee_position")
v1_rot_ee = RotationProperty.Config(label="ee_rotation")
v1_sca_ee = BoolProperty.Config(label="ee_scalar")

v1_pos_slide = PositionProperty.Config(label="slide_position")
v1_rot_slide = RotationProperty.Config(label="slide_rotation")
v1_sca_slide = RangeProperty.Config(label="slide_scalar", low=0.0, high=0.28)

v1_pos_drawer = PositionProperty.Config(label="drawer_position")
v1_rot_drawer = RotationProperty.Config(label="drawer_rotation")
v1_sca_drawer = RangeProperty.Config(label="drawer_scalar", low=0.0, high=0.22)

v1_pos_button = PositionProperty.Config(label="button_position")
v1_rot_button = RotationProperty.Config(label="button_rotation")
v1_sca_button = FlipProperty.Config(label="button_scalar")

v1_pos_led = PositionProperty.Config(label="led_position")
v1_rot_led = RotationProperty.Config(label="led_rotation")

v1_pos_block_red = AreaProperty.Config(label="block_red_position")
v1_sca_block_red = BoolProperty.Config(label="block_red_scalar")
v1_rot_block_red = RotationIgnoreProperty.Config(label="block_red_rotation")

v1_pos_block_blue = AreaProperty.Config(label="block_blue_position")
v1_sca_block_blue = BoolProperty.Config(label="block_blue_scalar")
v1_rot_block_blue = RotationIgnoreProperty.Config(label="block_blue_rotation")

v1_pos_block_pink = AreaProperty.Config(label="block_pink_position")
v1_sca_block_pink = BoolProperty.Config(label="block_pink_scalar")
v1_rot_block_pink = RotationIgnoreProperty.Config(label="block_pink_rotation")


base = [
    v1_pos_ee,
    v1_rot_ee,
    v1_sca_ee,
    v1_pos_drawer,
    v1_rot_drawer,
    v1_sca_drawer,
    v1_pos_button,
    v1_rot_button,
    v1_sca_button,
    v1_pos_led,
    v1_rot_led,
]
slide_base = [
    v1_pos_slide,
    v1_rot_slide,
    v1_sca_slide,
]
red_base = [
    v1_pos_block_red,
    v1_rot_block_red,
    v1_sca_block_red,
]
pink_base = [
    v1_pos_block_pink,
    v1_rot_block_pink,
    v1_sca_block_pink,
]
blue_base = [
    v1_pos_block_blue,
    v1_rot_block_blue,
    v1_sca_block_blue,
]

properties = base + slide_base + red_base + pink_base + blue_base
