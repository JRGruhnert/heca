from dataclasses import dataclass

from hoopgn.environments.environment import Environment
from hoopgn.misc.classes import SearchableClass
from hoopgn.properties.default.v2.domain import DomainProperty
from hoopgn.properties.default.v2.position import PositionProperty
from hoopgn.properties.default.v2.rotation import RotationProperty
from hoopgn.properties.default.v2.state import StateProperty


class Entity(SearchableClass):
    @dataclass(kw_only=True)
    class Query(SearchableClass.Query):
        env: Environment.Query

    @dataclass(kw_only=True)
    class Config(SearchableClass.Config):
        pass
