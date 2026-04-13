from dataclasses import dataclass

from hoopgn.plots.skill_plots.skill_plot import SkillPlot, SkillPlotConfig
from hoopgn.runners.skill.skill_runner import SkillRunner, SkillRunnerConfig
from hoopgn.skills.skill import Skill


@dataclass
class SkillPlotRunnerConfig(SkillRunnerConfig):
    plots: list[SkillPlotConfig]


class SkillPlotRunner(SkillRunner):
    def __init__(self, config: SkillPlotRunnerConfig):
        super().__init__(config)
        self.config = config
        self.plots = [SkillPlot(c) for c in config.plots]

    def run(self, skill: Skill):
        for plot in self.plots:
            plot(skill)
