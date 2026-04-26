from hoopgn.entities.entity import Entity
from hoopgn.properties.property import Property
from hoopgn.properties.default.v1.area import AreaProperty
from hoopgn.properties.default.v1.bool import BoolProperty
from hoopgn.properties.default.v1.flip import FlipProperty
from hoopgn.properties.default.v1.position import Position
from hoopgn.properties.default.v1.range import RangeProperty
from hoopgn.properties.default.v1.rotation import Rotation
from hoopgn.properties.evaluators.default import DefaultEvaluator
from hoopgn.environments.environment import Environment


ee_position = PositionConfig(
    query=Position.Query(
        label="ee_position",
        # id=0,
    ),
)
ee_rotation = RotationConfig(
    query=Property.Query(
        label="ee_rotation",
        # id=1,
    ),
)
ee_scalar = BoolPropertyConfig(
    query=Property.Query(
        label="ee_scalar",
        # id=2,
    ),
)
slide_position = PositionConfig(
    query=Property.Query(
        label="slide_position",
        # id=3,
    ),
)
slide_rotation = RotationConfig(
    query=Property.Query(
        label="slide_rotation",
        # id=4,
    ),
)
slide_scalar = RangePropertyConfig(
    query=Property.Query(
        label="slide_scalar",
        # id=5,
    ),
    low=0.0,
    high=0.28,
)
drawer_position = PositionConfig(
    query=Property.Query(
        label="drawer_position",
        # id=6,
    ),
)
drawer_rotation = RotationConfig(
    query=Property.Query(
        label="drawer_rotation",
        # id=7,
    ),
)
drawer_scalar = RangePropertyConfig(
    query=Property.Query(
        label="drawer_scalar",
        # id=8,
    ),
    low=0.0,
    high=0.22,
)
button_position = PositionConfig(
    query=Property.Query(
        label="button_position",
        # id=9,
    ),
)
button_rotation = RotationConfig(
    query=Property.Query(
        label="button_rotation",
        # id=10,
    ),
)
button_scalar = FlipPropertyConfig(
    query=Property.Query(
        label="button_scalar",
        # id=11,
    ),
)
led_position = PositionConfig(
    query=Property.Query(
        label="led_position",
        # id=12,
    ),
)
led_rotation = RotationConfig(
    query=Property.Query(
        label="led_rotation",
        # id=13,
    ),
)
block_red_position = AreaProperty.Query(
    label="block_red_position",
)


block_red_rotation = RotationConfig(
    query=Property.Query(
        label="block_red_rotation",
        # id=15,
    ),
    evaluator=DefaultEvaluator.Config(),
)
block_red_scalar = BoolPropertyConfig(
    query=Property.Query(
        label="block_red_scalar",
        # id=16,
    ),
)
block_blue_position = AreaPropertyConfig(
    query=Property.Query(
        label="block_blue_position",
        # id=17,
    ),
)
block_blue_rotation = RotationConfig(
    query=Property.Query(
        label="block_blue_rotation",
        # id=18,
    ),
    evaluator=DefaultEvaluator.Config(),
)
block_blue_scalar = BoolPropertyConfig(
    query=Property.Query(
        label="block_blue_scalar",
        # id=19,
    ),
)
block_pink_position = AreaPropertyConfig(
    query=Property.Query(
        label="block_pink_position",
        # id=20,
    ),
)
block_pink_rotation = RotationConfig(
    query=Property.Query(
        label="block_pink_rotation",
        # id=21,
    ),
    evaluator=DefaultEvaluator.Config(),
)
block_pink_scalar = BoolPropertyConfig(
    query=Property.Query(
        label="block_pink_scalar",
        # id=22,
    ),
)
