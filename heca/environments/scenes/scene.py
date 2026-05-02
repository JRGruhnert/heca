from abc import abstractmethod
from dataclasses import dataclass

import numpy as np
from heca.classes.register import Registerable
from heca.converters.converter import Converter
from heca.entities.entity import Entity
from heca.misc.td import TDScene


class Scene(Registerable):
    @dataclass(kw_only=True)
    class Config(Registerable.Config):
        converters: dict[str, Converter.Config]
        gt: bool = False

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.converters = {
            k: Converter.create(v) for k, v in self.cfg.converters.items()
        }

    def to_observation(self, obs) -> TDScene:
        formats = {k: self.converters[k](obs) for k in self.converters}
        return TDScene(formats=formats)

    @abstractmethod
    def step(self, action: np.ndarray) -> TDScene:
        raise NotImplementedError()

    @abstractmethod
    def close(self):
        raise NotImplementedError()

    @abstractmethod
    def sample(self) -> TDScene:
        raise NotImplementedError()

    def entities(self) -> list[Entity.Query]:
        raise NotImplementedError()

    def preuse(self) -> list[Entity.Query]:
        return self.entities()
