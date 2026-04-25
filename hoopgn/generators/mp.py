from dataclasses import dataclass

from hoopgn.agents.agent import Agent
from hoopgn.generators.generator import Generator
from hoopgn.misc.td import TDScene


class MPGenerator(Generator):
    @dataclass(kw_only=True)
    class Config(Generator.Config):
        agents: list[Agent.Config]

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg

    def __call__(
        self, x: TDScene, y: TDScene
    ) -> list[tuple[Agent.Query, TDScene, TDScene]]:
        temp = []
        for agent in self.cfg.agents:
            temp.append((agent, x, y))
        return temp
