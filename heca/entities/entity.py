from dataclasses import dataclass

from heca.classes.config import Configurable
from heca.properties.property import Property


class Entity(Configurable):
    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        label: str
        version: str
        props: set[Property.Config]
        weights: list[float] | None = None
        meta: str = "root"

    def __init__(self, cfg: Config):
        self.cfg = cfg

        weights = self.cfg.weights or [1.0 / len(self.cfg.props)] * len(self.cfg.props)
        self.properties: dict[str, tuple[Property, float]] = {
            v.label: (Property.create(v), w) for v, w in zip(self.cfg.props, weights)
        }
