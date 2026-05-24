from dataclasses import dataclass
from enum import Enum

from heca.classes.config import Configurable
from heca.properties.default.v2.position import PositionProperty
from heca.properties.default.v2.state import StateProperty


class Mobility(Enum):
    FREE = "free"  # Can move freely in the scene
    STATIC = "static"  # Has a fixed position and rotation in the scene
    ARTICULATED = "articulated"  # Has a fixed position but can have a variable rotation in the scene


class Entity(Configurable):
    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        label: str
        states: set[str]
        mobility: Mobility
        position: PositionProperty.Config = PositionProperty.Config()
        rotation: PositionProperty.Config = PositionProperty.Config()

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.position = PositionProperty.create(cfg.position)
        self.rotation = PositionProperty.create(cfg.rotation)
        self.state = StateProperty.create(
            StateProperty.Config(
                values=cfg.states,
            )
        )
