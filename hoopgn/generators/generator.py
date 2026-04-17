from dataclasses import dataclass

from hoopgn.base import ConfigurableClass


class Hoopgn(ConfigurableClass):
    @dataclass(kw_only=True)
    class Config(ConfigurableClass.Config):
        pass

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def reset(self, goal):
        pass

    def __call__(self, x):
        raise NotImplementedError
