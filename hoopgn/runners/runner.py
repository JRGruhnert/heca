from abc import abstractmethod
from dataclasses import dataclass

from hoopgn.classes import ConfigurableClass

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
        assert isinstance(self.agent, BranchAgent)
        self.agent.train(10)

    def plot(self):
        for plotter in self.plotters:
            plotter.plot()

    def explain(self):
        obs, goal = self.experiment.sample_task()
        episode_ended = False
        while not episode_ended:
            skill = self.agent.act(obs, goal)
            actor_expl, critic_expl = self.agent.explain(obs, goal, skill)
