from abc import abstractmethod
from dataclasses import dataclass

from heca.plotters.plotter import Plotter, PlotterConfig
from heca.agents.agent import Agent


@dataclass
class SkillPlotterConfig(PlotterConfig):
    pass


class SkillPlotter(Plotter):
    def __init__(self, config: SkillPlotterConfig):
        super().__init__(config)
        self.config = config

    @abstractmethod
    def plot_content(self, skill: Agent):
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        raise NotImplementedError()
