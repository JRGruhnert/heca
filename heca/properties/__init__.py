from heca.properties.default.v1.area import AreaProperty
from heca.properties.default.v1.bool import BoolProperty
from heca.properties.default.v1.flip import FlipProperty
from heca.properties.default.v1.position import PositionProperty
from heca.properties.default.v1.range import RangeProperty
from heca.properties.default.v1.rotation import RotationProperty
from heca.properties.evaluators.default import DefaultEvaluator

ee_position = PositionProperty.Config(label="ee_position")
ee_rotation = RotationProperty.Config(label="ee_rotation")
ee_scalar = BoolProperty.Config(label="ee_scalar")
slide_position = PositionProperty.Config(label="slide_position")
slide_rotation = RotationProperty.Config(label="slide_rotation")
slide_scalar = RangeProperty.Config(label="slide_scalar", low=0.0, high=0.28)
drawer_position = PositionProperty.Config(label="drawer_position")
drawer_rotation = RotationProperty.Config(label="drawer_rotation")
drawer_scalar = RangeProperty.Config(label="drawer_scalar", low=0.0, high=0.22)
button_position = PositionProperty.Config(label="button_position")
button_rotation = RotationProperty.Config(label="button_rotation")
button_scalar = FlipProperty.Config(label="button_scalar")
led_position = PositionProperty.Config(label="led_position")
led_rotation = RotationProperty.Config(label="led_rotation")
block_red_position = AreaProperty.Config(label="block_red_position")
block_red_scalar = BoolProperty.Config(label="block_red_scalar")
block_blue_position = AreaProperty.Config(label="block_blue_position")
block_blue_scalar = BoolProperty.Config(label="block_blue_scalar")
block_pink_position = AreaProperty.Config(label="block_pink_position")
block_pink_scalar = BoolProperty.Config(label="block_pink_scalar")
block_red_rotation = RotationProperty.Config(
    label="block_red_rotation",
    evaluator=DefaultEvaluator.Config(),
)
block_blue_rotation = RotationProperty.Config(
    label="block_blue_rotation",
    evaluator=DefaultEvaluator.Config(),
)
block_pink_rotation = RotationProperty.Config(
    label="block_pink_rotation",
    evaluator=DefaultEvaluator.Config(),
)


base = [
    ee_position,
    ee_rotation,
    ee_scalar,
    drawer_position,
    drawer_rotation,
    drawer_scalar,
    button_position,
    button_rotation,
    button_scalar,
    led_position,
    led_rotation,
]

slide_base = [
    slide_position,
    slide_rotation,
    slide_scalar,
]

red_base = [
    block_red_position,
    block_red_rotation,
    block_red_scalar,
]

pink_base = [
    block_pink_position,
    block_pink_rotation,
    block_pink_scalar,
]

blue_base = [
    block_blue_position,
    block_blue_rotation,
    block_blue_scalar,
]
