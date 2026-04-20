from dataclasses import dataclass

from hoopgn.base import ConfigurableClass
from hoopgn.observation.td_scene import TDScene


class Hoopgn(ConfigurableClass):
    @dataclass(kw_only=True)
    class Config(ConfigurableClass.Config):
        pass

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def reset(self, goal: TDScene):
        pass

    def __call__(self, x: TDScene) -> TDScene:
        raise NotImplementedError
