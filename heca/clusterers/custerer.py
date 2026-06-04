from dataclasses import dataclass

from heca.agents.agent import Agent
from heca.misc.base import Configurable
from heca.entities.precon import Precon
from heca.misc.td import TDScene


class Compressor(Configurable):
    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        pass

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def __call__(self, x: TDScene, y: TDScene) -> TDScene:
        raise NotImplementedError()

    def cluster(
        self, precons: list[Precon], postcons: list[Postcon]
    ) -> list[list[Entity]]:
        raise NotImplementedError()

    def merge(self, clusters: list[list[Entity]]) -> list[Entity]:
        raise NotImplementedError()
