from hoopgn.entities.entity import Entity
from hoopgn.entities.properties.property import Property
from hoopgn.entities.properties.default.v2.domain import DomainConfig
from hoopgn.entities.properties.default.v2.position import PositionConfig
from hoopgn.entities.properties.default.v2.rotation import RotationConfig
from hoopgn.entities.properties.default.v2.state import StateConfig


red = Entity.Config(
    query=Entity.Query(
        label="red",
        id=0,
    ),
    domain=DomainConfig(
        query=Property.Query(
            label="domain",
            # id=0,
        ),
    ),
    position=PositionConfig(
        query=Property.Query(
            label="block_red_position",
            # id=0,
        ),
    ),
    rotation=RotationConfig(
        query=Property.Query(
            label="block_red_rotation",
            # id=1,
        ),
    ),
    state=StateConfig(
        query=Property.Query(
            label="block_red_scalar",
            # id=2,
        ),
    ),
)
blue = Entity.Config(
    query=Entity.Query(
        label="blue",
        id=1,
    ),
    domain=DomainConfig(
        query=Property.Query(
            label="domain",
            # id=0,
        ),
    ),
    position=PositionConfig(
        query=Property.Query(
            label="block_blue_position",
            # id=0,
        ),
    ),
    rotation=RotationConfig(
        query=Property.Query(
            label="block_blue_rotation",
            # id=1,
        ),
    ),
    state=StateConfig(
        query=Property.Query(
            label="block_blue_scalar",
            # id=2,
        ),
    ),
)
pink = Entity.Config(
    query=Entity.Query(
        label="pink",
        id=2,
    ),
    domain=DomainConfig(
        query=Property.Query(
            label="domain",
            # id=0,
        ),
    ),
    position=PositionConfig(
        query=Property.Query(
            label="block_pink_position",
            # id=0,
        ),
    ),
    rotation=RotationConfig(
        query=Property.Query(
            label="block_pink_rotation",
            # id=1,
        ),
    ),
    state=StateConfig(
        query=Property.Query(
            label="block_pink_scalar",
            # id=2,
        ),
    ),
)
slider = Entity.Config(
    query=Entity.Query(
        label="slider",
        id=3,
    ),
    domain=DomainConfig(
        query=Property.Query(
            label="domain",
            # id=0,
        ),
    ),
    position=PositionConfig(
        query=Property.Query(
            label="slide_position",
            # id=0,
        ),
    ),
    rotation=RotationConfig(
        query=Property.Query(
            label="slide_rotation",
            # id=1,
        ),
    ),
    state=StateConfig(
        query=Property.Query(
            label="slide_scalar",
            # id=2,
        ),
    ),
)
drawer = Entity.Config(
    query=Entity.Query(
        label="drawer",
        id=4,
    ),
    domain=DomainConfig(
        query=Property.Query(
            label="domain",
            # id=0,
        ),
    ),
    position=PositionConfig(
        query=Property.Query(
            label="drawer_position",
            # id=0,
        ),
    ),
    rotation=RotationConfig(
        query=Property.Query(
            label="drawer_rotation",
            # id=1,
        ),
    ),
    state=StateConfig(
        query=Property.Query(
            label="drawer_scalar",
            # id=2,
        ),
    ),
)
button = Entity.Config(
    query=Entity.Query(
        label="button",
        id=5,
    ),
    domain=DomainConfig(
        query=Property.Query(
            label="domain",
            # id=0,
        ),
    ),
    position=PositionConfig(
        query=Property.Query(
            label="button_position",
            # id=0,
        ),
    ),
    rotation=RotationConfig(
        query=Property.Query(
            label="button_rotation",
            # id=1,
        ),
    ),
    state=StateConfig(
        query=Property.Query(
            label="button_scalar",
            # id=2,
        ),
    ),
)
led = Entity.Config(
    query=Entity.Query(
        label="led",
        id=6,
    ),
    domain=DomainConfig(
        query=Property.Query(
            label="domain",
            # id=0,
        ),
    ),
    position=PositionConfig(
        query=Property.Query(
            label="led_position",
            # id=0,
        ),
    ),
    rotation=RotationConfig(
        query=Property.Query(
            label="led_rotation",
            # id=1,
        ),
    ),
    state=StateConfig(
        query=Property.Query(
            label="button_scalar",
            # id=2,
        ),
    ),
)
