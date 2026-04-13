from dataclasses import dataclass

from hoopgn.plotters.plots.entity_3d import (
    Entity3DHelper,
    Entity3DHelperConfig,
    Entity3DHelperConfig,
    Entity3DMode,
)
from hoopgn.plotters.skill_plotters.skill_plotter import (
    SkillPlotter,
    SkillPlotterConfig,
)
from hoopgn.skills.skill import Skill


@dataclass
class SkillConditionsPlotConfig(SkillPlotterConfig):
    name: str = "SkillConditions3D"
    entity3d: Entity3DHelperConfig = Entity3DHelperConfig(
        mode=Entity3DMode.COLORS,
    )


class SkillConditionsPlot(SkillPlotter):
    def __init__(self, config: SkillConditionsPlotConfig):
        super().__init__(config)
        self.config = config
        self.entity3d = Entity3DHelper(config.entity3d)

    def plot_content(self, skill: Skill):
        for precons in skill.precons:
            self.entity3d.set_entity(
                current=precons.current,
                goal=precons.goal,
            )
        self.entity3d.finish()
