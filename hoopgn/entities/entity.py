from dataclasses import dataclass

from hoopgn.base import RegisterableClass
from hoopgn.properties.v2.domain import DomainConfig
from hoopgn.properties.v2.position import PositionConfig
from hoopgn.properties.v2.rotation import RotationConfig
from hoopgn.properties.v2.state import StateConfig


class Entity(RegisterableClass):
    @dataclass(kw_only=True)
    class Signature(RegisterableClass.Signature):
        label: str
        id: int

    @dataclass(kw_only=True)
    class Config(RegisterableClass.Config):
        state: StateConfig
        domain: DomainConfig
        position: PositionConfig
        rotation: RotationConfig

    def __init__(self, cfg: Config):
        self.cfg = cfg
