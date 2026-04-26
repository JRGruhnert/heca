from dataclasses import dataclass

from hoopgn.entities.entity import Entity
from hoopgn.environments.environment import Environment
from hoopgn.properties.default.v2.domain import DomainProperty
from hoopgn.properties.default.v2.position import PositionProperty
from hoopgn.properties.default.v2.rotation import RotationProperty
from hoopgn.properties.default.v2.state import StateProperty


class EntityV2(Entity):
    @dataclass(kw_only=True)
    class Query(Entity.Query):
        env: Environment.Query

    @dataclass(kw_only=True)
    class Config(Entity.Config):
        state: StateProperty.Config
        domain: DomainProperty.Config
        position: PositionProperty.Config
        rotation: RotationProperty.Config
