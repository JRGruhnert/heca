from abc import abstractmethod
from dataclasses import dataclass

import numpy as np
from heca.misc.td import TDScene
from heca.classes.register import Registerable
from heca.converters.converter import HecaConverter, LeafConverter


class Environment(Registerable):
    @dataclass(kw_only=True)
    class Config(Registerable.Config):
        heca_cv: HecaConverter.Config
        leaf_cv: LeafConverter.Config

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.heca_cv = HecaConverter.create(self.cfg.heca_cv)
        self.leaf_cv = LeafConverter.create(self.cfg.leaf_cv)

    def to_observation(self, obs) -> TDScene:
        heca = self.heca_cv(obs)
        leaf = self.leaf_cv(obs)
        return TDScene(heca=heca, leaf=leaf)

    @abstractmethod
    def sample(self) -> TDScene:
        raise NotImplementedError()

    @abstractmethod
    def step(self, action: np.ndarray) -> TDScene:
        raise NotImplementedError()

    @abstractmethod
    def render(self):
        raise NotImplementedError()

    @abstractmethod
    def close(self):
        raise NotImplementedError()
