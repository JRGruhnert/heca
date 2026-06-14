from dataclasses import dataclass
from enum import Enum
from textwrap import dedent

from heca.misc.base import Configurable
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
        question: str
        answers: set[str]
        mobility: Mobility
        position: PositionProperty.Config = PositionProperty.Config()
        rotation: PositionProperty.Config = PositionProperty.Config()

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.position = PositionProperty.get(cfg.position)
        self.rotation = PositionProperty.get(cfg.rotation)
        self.state = StateProperty.get(
            StateProperty.Config(
                values=cfg.states,
            )
        )
