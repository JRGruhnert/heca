from abc import abstractmethod
from dataclasses import dataclass

from hoopgn.agents.hoop import HoopAgent
from hoopgn.misc.classes import ConfigurableClass

from hoopgn.agents.agent import Agent
from hoopgn.plotters.hoopgn_plotters import select_multiple_hoopgn_plotters
from hoopgn.plotters.hoopgn_plotters.hoopgn_plotter import HoopGNPlotterConfig


class HoopGNRunner(ConfigurableClass):
    @dataclass(kw_only=True)
    class Config(ConfigurableClass.Config):
        agent: Agent.Config
        plotters: list[HoopGNPlotterConfig]
        epochs: int = 100

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.agent = Agent.from_config(cfg.agent)
        self.plotters = select_multiple_hoopgn_plotters(cfg.plotters)

    def train(self):
        assert isinstance(self.agent, HoopAgent)
        self.agent.train(10)

    def plot(self):
        for plotter in self.plotters:
            plotter.plot()

    def explain(self):
        assert isinstance(self.agent, HoopAgent)
        obs, goal = self.agent.sample()
        episode_ended = False
        while not episode_ended:
            self.agent.explain()
