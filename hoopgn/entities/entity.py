from dataclasses import dataclass
from hoopgn.base import RegisterableClass
from hoopgn.properties.positions.position import PositionConfig
from hoopgn.properties.property import Property


class Entity(RegisterableClass):
    @dataclass(kw_only=True)
    class Signature(RegisterableClass.Signature):
        id: int
        label: str

    @dataclass(kw_only=True)
    class Config(RegisterableClass.Config):
        domain: Property.Config
        position: PositionConfig
        rotation: Property.Config
        state: Property.Config

    def __init__(self, cfg: Config):
        self.cfg = cfg
