from hoopgn.entities.entity import Entity, EntityConfig

from hoopgn.entities.entity import Entity
from hoopgn.observation.td_entity import TDEntity
from hoopgn.observation.td_properties import TDProperties
from hoopgn.properties.property import Property

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


def properties_to_entities(properties: list[Property.Config]) -> list[Entity.Config]:
    property_dict = {prop.label: prop for prop in properties}
    entities: list[Entity.Config] = []
    for entity_name, props in _ENTITY_PROPERTY_MAPPING.items():
        if (
            props["position"] not in property_dict
            or props["rotation"] not in property_dict
            or props["scalar"] not in property_dict
        ):
            continue

        cfg = Entity.Config(
            id=len(entities),
            label=entity_name,
            position=property_dict[props["position"]],
            rotation=property_dict[props["rotation"]],
            state=property_dict[props["scalar"]],
        )
        entities.append(cfg)
    return entities


def tdp_to_tde(properties: TDProperties, entity: Entity) -> TDEntity:
    return TDEntity(
        position=properties[entity.config.position.label],
        rotation=properties[entity.config.rotation.label],
        state=properties[entity.config.state.label],
    )


def get_entity_set(tag: str) -> list[EntityConfig]:
    raise NotImplementedError()


def select_entities(configs: list[EntityConfig]) -> list[Entity]:
    """Create entities from configs - simple factory function"""
    return [Entity(config) for config in configs]
