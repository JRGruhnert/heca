from dataclasses import dataclass

from heca.misc.classes import Configurable
from heca.misc.td import TDScene


class Clusterer(Configurable):
    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        pass

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def __call__(self, x: TDScene, y: TDScene) -> TDScene:
        raise NotImplementedError()
