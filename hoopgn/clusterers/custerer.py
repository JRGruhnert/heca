from dataclasses import dataclass

from hoopgn.misc.classes import ConfigClass
from hoopgn.misc.td import TDScene


class Clusterer(ConfigClass):
    @dataclass(kw_only=True)
    class Config(ConfigClass.Config):
        pass

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def __call__(self, x: TDScene, y: TDScene) -> TDScene:
        raise NotImplementedError()
