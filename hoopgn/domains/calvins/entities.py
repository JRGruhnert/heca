from hoopgn.entities.entity import Entity
from hoopgn.properties.property import Property
from hoopgn.properties.v2.domain import DomainPropertyConfig
from hoopgn.properties.v2.position import PositionPropertyConfig
from hoopgn.properties.v2.rotation import RotationPropertyConfig
from hoopgn.properties.v2.state import StatePropertyConfig


red = Entity.Config(
    sig=Entity.Signature(
        label="red",
        sig
    ),
    domain=DomainPropertyConfig(
        sig=Property.Signature(
            label="calvins",
            id=0,
        ),
    ),
    position=PositionPropertyConfig(
        sig=Property.Signature(
            label="block_red_position",
            id=0,
        ),
    ),
    rotation=RotationPropertyConfig(
        sig=Property.Signature(
            label="block_red_rotation",
            id=1,
        ),
    ),
    state=StatePropertyConfig(
        sig=Property.Signature(
            label="block_red_scalar",
            id=2,
        ),
    ),
)
_ENTITY_PROPERTY_MAPPING: dict[str, dict[str, str]] = {
    "red": {
        "position": "block_red_position",
        "rotation": "block_red_rotation",
        "scalar": "block_red_scalar",
    },
    "blue": {
        "position": "block_blue_position",
        "rotation": "block_blue_rotation",
        "scalar": "block_blue_scalar",
    },
    "pink": {
        "position": "block_pink_position",
        "rotation": "block_pink_rotation",
        "scalar": "block_pink_scalar",
    },
    "slider": {
        "position": "slide_position",
        "rotation": "slide_rotation",
        "scalar": "slide_scalar",
    },
    "drawer": {
        "position": "drawer_position",
        "rotation": "drawer_rotation",
        "scalar": "drawer_scalar",
    },
    "button": {
        "position": "button_position",
        "rotation": "button_rotation",
        "scalar": "button_scalar",
    },
    "led": {
        "position": "led_position",
        "rotation": "led_rotation",
        "scalar": "button_scalar",
    },
}
