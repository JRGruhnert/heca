from dataclasses import dataclass

from hoopgn.agents.agent import Agent
from hoopgn.misc.classes import ConfigurableClass
from hoopgn.misc.td import TDScene
from torch_geometric.data import Batch


class Generator(ConfigurableClass):
    @dataclass(kw_only=True)
    class Config(ConfigurableClass.Config):
        pass

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def __call__(
        self, x: TDScene, y: TDScene
    ) -> tuple[list[tuple[Agent.Query, TDScene, TDScene]], Batch]:
        raise NotImplementedError()
