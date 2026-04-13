from dataclasses import dataclass

from hoopgn.plotters.skill_plotters import select_skill_plotter
from hoopgn.plotters.skill_plotters.skill_plotter import (
    SkillPlotter,
    SkillPlotterConfig,
)
from hoopgn.runners.skill.skill_runner import SkillRunner, SkillRunnerConfig
from hoopgn.skills.skill import Skill, SkillConfig


@dataclass
class SkillPlotRunnerConfig(SkillRunnerConfig):
    plotters: list[SkillPlotterConfig]
    skill: SkillConfig | None = None


class SkillPlotRunner(SkillRunner):
    def __init__(self, config: SkillPlotRunnerConfig):
        super().__init__(config)
        self.config = config
        self.plotters = [select_skill_plotter(c) for c in config.plotters]

    def skill_run(self, skill: Skill):
        for plotter in self.plotters:
            plotter.plot_content(skill)
