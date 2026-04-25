from dataclasses import dataclass

from hoopgn.classes import ConfigurableClass
from hoopgn.misc.td import TDScene


class Clusterer(ConfigurableClass):
    @dataclass(kw_only=True)
    class Config(ConfigurableClass.Config):
        pass

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def __call__(self, x: TDScene, y: TDScene) -> TDScene:
        raise NotImplementedError()
