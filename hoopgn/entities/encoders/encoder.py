from dataclasses import dataclass

from hoopgn.misc.classes import StoragableClass
from hoopgn.misc.td import TDEntity
from hoopgn.properties.encoders.encoder import PropertyEncoder
from hoopgn.properties.encoders.v2.domain import DomainEncoderConfig
from hoopgn.properties.encoders.v2.position import PositionEncoderConfig
from hoopgn.properties.encoders.v2.rotation import QuaternionEncoderConfig
from hoopgn.properties.encoders.v2.state import StateEncoderConfig


class EntityEncoder(StoragableClass):
    @dataclass(kw_only=True)
    class Query(StoragableClass.Query):
        label: str

    @dataclass(kw_only=True)
    class Config(StoragableClass.Config):
        domain: DomainEncoderConfig
        position: PositionEncoderConfig
        rotation: QuaternionEncoderConfig
        state: StateEncoderConfig

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.domain = PropertyEncoder.search(cfg.domain.query)
        self.position = PropertyEncoder.search(cfg.position.query)
        self.rotation = PropertyEncoder.search(cfg.rotation.query)
        self.state = PropertyEncoder.search(cfg.state.query)

    def forward(self, x: TDEntity):
        x.domain = self.domain(x.domain)
        x.position = self.position(x.position)
        x.rotation = self.rotation(x.rotation)
        x.state = self.state(x.state)
        return x
