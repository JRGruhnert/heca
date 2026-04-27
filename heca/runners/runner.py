from abc import abstractmethod
from dataclasses import dataclass

from heca.agents.hecas.heca import Heca
from heca.misc.classes import ConfigClass

from heca.agents.agent import Agent
from heca.plotters.hoopgn_plotters import select_multiple_hoopgn_plotters
from heca.plotters.hoopgn_plotters.hoopgn_plotter import HoopGNPlotterConfig


class HoopGNRunner(ConfigClass):
    @dataclass(kw_only=True)
    class Config(ConfigClass.Config):
        agent: Agent.Config
        plotters: list[HoopGNPlotterConfig]
        epochs: int = 100

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.agent = Agent.from_config(cfg.agent)
        self.plotters = select_multiple_hoopgn_plotters(cfg.plotters)

    def train(self):
        assert isinstance(self.agent, Heca)
        self.agent.train(10)

    def plot(self):
        for plotter in self.plotters:
            plotter.plot()

    def explain(self):
        assert isinstance(self.agent, Heca)
        obs, goal = self.agent.sample()
        episode_ended = False
        while not episode_ended:
            self.agent.explain()
