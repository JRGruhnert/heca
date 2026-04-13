from abc import abstractmethod
from dataclasses import dataclass

from hoopgn.plotters.plotter import Plotter, PlotterConfig
from hoopgn.skills.skill import Skill


@dataclass
class SkillPlotterConfig(PlotterConfig):
    pass


class SkillPlotter(Plotter):
    def __init__(self, config: SkillPlotterConfig):
        super().__init__(config)
        self.config = config

    @abstractmethod
    def plot_content(self, skill: Skill):
        raise NotImplementedError()
