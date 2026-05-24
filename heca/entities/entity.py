from dataclasses import dataclass

from heca.classes.config import Configurable

from heca.properties.default.v2.position import PositionProperty
from heca.properties.default.v2.state import StateProperty


class Entity(Configurable):
    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        label: str
        states: set[str]
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
