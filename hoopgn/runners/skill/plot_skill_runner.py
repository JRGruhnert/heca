from dataclasses import dataclass

from hoopgn.plots.skill_plots.skill_plot import SkillPlot, SkillPlotConfig
from hoopgn.runners.skill.skill_runner import SkillRunner, SkillRunnerConfig
from hoopgn.skills.skill import Skill


@dataclass
class SkillPlotterConfig(SkillRunnerConfig):
    plots: list[SkillPlotConfig]


class SkillPlotter(SkillRunner):
    def __init__(self, config: SkillPlotterConfig):
        super().__init__(config)
        self.config = config
        self.plots = [SkillPlot(c) for c in config.plots]

    def run(self, skill: Skill):
        for plot in self.plots:
            plot(skill)
