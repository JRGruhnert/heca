from heca.entities.entity import Entity
from heca.properties.default.v2.position import PositionProperty
from heca.properties.default.v2.rotation import RotationProperty
from heca.properties.default.v2.state import StateProperty, State

CLAVIN_ENTITIES = {
    Entity.Query(label="ee", meta="ee", env="calvin"): Entity.Config(
        props=set(
            [
                PositionProperty.Config(label="ee_position"),
                RotationProperty.Config(label="ee_rotation"),
                StateProperty.Config(
                    label="ee_scalar",
                    state=State.Config(values={"on", "off"}),
                ),
            ]
        ),
        weights=list(),
    )
}


entities = [CLAVIN_ENTITIES[Entity.Query(label="ee", meta="ee", env="calvin")]]
