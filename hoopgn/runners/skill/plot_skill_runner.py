from dataclasses import dataclass, field

from hoopgn.properties.property import Property
from hoopgn.entities.entity import Entity
from hoopgn.plotters.skill_plotters import select_skill_plotter
from hoopgn.plotters.skill_plotters.skill_plotter import (
    SkillPlotter,
    SkillPlotterConfig,
)
from hoopgn.runners.skill.skill_runner import SkillRunner, SkillRunnerConfig
from hoopgn.skills.skill import Skill, SkillConfig


@dataclass
class SkillPlotRunnerConfig(SkillRunnerConfig):
    plotters: list[SkillPlotterConfig] = field(default_factory=list)
    skill: SkillConfig | None = None


class SkillPlotRunner(SkillRunner):
    def __init__(self, config: SkillPlotRunnerConfig):
        super().__init__(config)
        self.config = config
        self.plotters = [select_skill_plotter(c) for c in config.plotters]
        self.entities = [Entity(e) for e in config.entities]
        self.properties = [Property(p) for p in config.properties]

    def skill_run(self, skill: Skill):
        for plotter in self.plotters:
            plotter.init(self.skills, self.entities, self.properties)
            plotter.plot_content(skill)
