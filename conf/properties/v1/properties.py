from conf.properties.v1.area import AreaPropertyConfig
from conf.properties.v1.bool import BoolPropertyConfig
from conf.properties.v1.flip import FlipPropertyConfig
from conf.properties.v1.range import RangePropertyConfig
from conf.properties.v1.position import PositionPropertyConfig
from conf.properties.v1.quaternion import QuaternionPropertyConfig
from hoopgn.environments.properties.features.evaluators import PIgnoreEvaluatorConfig


ee_position: PositionPropertyConfig = PositionPropertyConfig(
    label="ee_position",
    id=0,
)
ee_rotation: QuaternionPropertyConfig = QuaternionPropertyConfig(
    label="ee_rotation",
    id=1,
)
ee_scalar: BoolPropertyConfig = BoolPropertyConfig(
    label="ee_scalar",
    id=2,
)
slide_position: PositionPropertyConfig = PositionPropertyConfig(
    label="slide_position",
    id=3,
)
slide_rotation: QuaternionPropertyConfig = QuaternionPropertyConfig(
    label="slide_rotation",
    id=4,
)
slide_scalar: RangePropertyConfig = RangePropertyConfig(
    label="slide_scalar",
    id=5,
    low=0.0,
    high=0.28,
)
drawer_position: PositionPropertyConfig = PositionPropertyConfig(
    label="drawer_position",
    id=6,
)
drawer_rotation: QuaternionPropertyConfig = QuaternionPropertyConfig(
    label="drawer_rotation",
    id=7,
)
drawer_scalar: RangePropertyConfig = RangePropertyConfig(
    label="drawer_scalar",
    id=8,
    low=0.0,
    high=0.22,
)
button_position: PositionPropertyConfig = PositionPropertyConfig(
    label="button_position",
    id=9,
)
button_rotation: QuaternionPropertyConfig = QuaternionPropertyConfig(
    label="button_rotation",
    id=10,
)
button_scalar: FlipPropertyConfig = FlipPropertyConfig(
    label="button_scalar",
    id=11,
)
led_position: PositionPropertyConfig = PositionPropertyConfig(
    label="led_position",
    id=12,
)
led_rotation: QuaternionPropertyConfig = QuaternionPropertyConfig(
    label="led_rotation",
    id=13,
)
block_red_position: AreaPropertyConfig = AreaPropertyConfig(
    label="block_red_position",
    id=14,
)
block_red_rotation: QuaternionPropertyConfig = QuaternionPropertyConfig(
    label="block_red_rotation",
    id=15,
    evaluator=PIgnoreEvaluatorConfig(),
)
block_red_scalar: BoolPropertyConfig = BoolPropertyConfig(
    label="block_red_scalar",
    id=16,
)
block_blue_position: AreaPropertyConfig = AreaPropertyConfig(
    label="block_blue_position",
    id=17,
)
block_blue_rotation: QuaternionPropertyConfig = QuaternionPropertyConfig(
    label="block_blue_rotation",
    id=18,
    evaluator=PIgnoreEvaluatorConfig(),
)
block_blue_scalar: BoolPropertyConfig = BoolPropertyConfig(
    label="block_blue_scalar",
    id=19,
)
block_pink_position: AreaPropertyConfig = AreaPropertyConfig(
    label="block_pink_position",
    id=20,
)
block_pink_rotation: QuaternionPropertyConfig = QuaternionPropertyConfig(
    label="block_pink_rotation",
    id=21,
    evaluator=PIgnoreEvaluatorConfig(),
)
block_pink_scalar: BoolPropertyConfig = BoolPropertyConfig(
    label="block_pink_scalar",
    id=22,
)
