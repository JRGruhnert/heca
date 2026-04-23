from hoopgn.properties.property import Property
from hoopgn.properties.default.v1.area import CalvinAreaPropertyConfig
from hoopgn.properties.default.v1.bool import BoolPropertyConfig
from hoopgn.properties.default.v1.flip import FlipPropertyConfig
from hoopgn.properties.default.v1.position import PositionConfig
from hoopgn.properties.default.v1.range import RangePropertyConfig
from hoopgn.properties.default.v1.rotation import RotationConfig
from hoopgn.properties.features.evaluators.default import DefaultEvaluator


ee_position = PositionConfig(
    sig=Property.Signature(
        label="ee_position",
        id=0,
    ),
)
ee_rotation = RotationConfig(
    sig=Property.Signature(
        label="ee_rotation",
        id=1,
    ),
)
ee_scalar = BoolPropertyConfig(
    sig=Property.Signature(
        label="ee_scalar",
        id=2,
    ),
)
slide_position = PositionConfig(
    sig=Property.Signature(
        label="slide_position",
        id=3,
    ),
)
slide_rotation = RotationConfig(
    sig=Property.Signature(
        label="slide_rotation",
        id=4,
    ),
)
slide_scalar = RangePropertyConfig(
    sig=Property.Signature(
        label="slide_scalar",
        id=5,
    ),
    low=0.0,
    high=0.28,
)
drawer_position = PositionConfig(
    sig=Property.Signature(
        label="drawer_position",
        id=6,
    ),
)
drawer_rotation = RotationConfig(
    sig=Property.Signature(
        label="drawer_rotation",
        id=7,
    ),
)
drawer_scalar = RangePropertyConfig(
    sig=Property.Signature(
        label="drawer_scalar",
        id=8,
    ),
    low=0.0,
    high=0.22,
)
button_position = PositionConfig(
    sig=Property.Signature(
        label="button_position",
        id=9,
    ),
)
button_rotation = RotationConfig(
    sig=Property.Signature(
        label="button_rotation",
        id=10,
    ),
)
button_scalar = FlipPropertyConfig(
    sig=Property.Signature(
        label="button_scalar",
        id=11,
    ),
)
led_position = PositionConfig(
    sig=Property.Signature(
        label="led_position",
        id=12,
    ),
)
led_rotation = RotationConfig(
    sig=Property.Signature(
        label="led_rotation",
        id=13,
    ),
)
block_red_position = CalvinAreaPropertyConfig(
    sig=Property.Signature(
        label="block_red_position",
        id=14,
    ),
)
block_red_rotation = RotationConfig(
    sig=Property.Signature(
        label="block_red_rotation",
        id=15,
    ),
    evaluator=DefaultEvaluator.Config(),
)
block_red_scalar = BoolPropertyConfig(
    sig=Property.Signature(
        label="block_red_scalar",
        id=16,
    ),
)
block_blue_position = CalvinAreaPropertyConfig(
    sig=Property.Signature(
        label="block_blue_position",
        id=17,
    ),
)
block_blue_rotation = RotationConfig(
    sig=Property.Signature(
        label="block_blue_rotation",
        id=18,
    ),
    evaluator=DefaultEvaluator.Config(),
)
block_blue_scalar = BoolPropertyConfig(
    sig=Property.Signature(
        label="block_blue_scalar",
        id=19,
    ),
)
block_pink_position = CalvinAreaPropertyConfig(
    sig=Property.Signature(
        label="block_pink_position",
        id=20,
    ),
)
block_pink_rotation = RotationConfig(
    sig=Property.Signature(
        label="block_pink_rotation",
        id=21,
    ),
    evaluator=DefaultEvaluator.Config(),
)
block_pink_scalar = BoolPropertyConfig(
    sig=Property.Signature(
        label="block_pink_scalar",
        id=22,
    ),
)
