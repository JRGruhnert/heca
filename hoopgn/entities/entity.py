from dataclasses import dataclass

from hoopgn.misc.classes import SearchableClass
from hoopgn.entities.properties.default.v2.domain import DomainConfig
from hoopgn.entities.properties.default.v2.position import PositionConfig
from hoopgn.entities.properties.default.v2.rotation import RotationConfig
from hoopgn.entities.properties.default.v2.state import StateConfig


class Entity(SearchableClass):
    @dataclass(kw_only=True)
    class Query(SearchableClass.Query):
        label: str
        id: int

    @dataclass(kw_only=True)
    class Config(SearchableClass.Config):
        state: StateConfig
        domain: DomainConfig
        position: PositionConfig
        rotation: RotationConfig

    def __init__(self, cfg: Config):
        self.cfg = cfg
