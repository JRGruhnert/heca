from dataclasses import dataclass

from heca.misc.classes import ConfigClass
from heca.misc.td import TDScene


class Clusterer(ConfigClass):
    @dataclass(kw_only=True)
    class Config(ConfigClass.Config):
        pass

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def __call__(self, x: TDScene, y: TDScene) -> TDScene:
        raise NotImplementedError()
