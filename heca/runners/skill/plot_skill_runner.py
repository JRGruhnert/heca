from dataclasses import dataclass, field

from heca.properties.property import Property
from heca.entities.entity import Entity
from heca.plotters.skill_plotters import select_skill_plotter
from heca.plotters.skill_plotters.skill_plotter import (
    SkillPlotter,
    SkillPlotterConfig,
)
from heca.runners.skill.skill_runner import SkillRunner, SkillRunnerConfig
from heca.agents.agent import Agent, SkillConfig


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

    def skill_run(self, skill: Agent):
        for plotter in self.plotters:
            plotter.init(self.agents, self.entities, self.properties)
            plotter.plot(skill)
            plotter.reset()
