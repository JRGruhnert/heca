from hoopgn.plotters.skill_plotters.cnd_plotter import (
    SkillConditionsPlot,
    SkillConditionsPlotterConfig,
)
from hoopgn.plotters.skill_plotters.skill_plotter import (
    SkillPlotterConfig,
    SkillPlotter,
)


def select_multiple_skill_plotters(
    configs: list[SkillPlotterConfig],
) -> list[SkillPlotter]:
    return [select_skill_plotter(c) for c in configs]


def select_skill_plotter(config: SkillPlotterConfig) -> SkillPlotter:
    if isinstance(config, SkillConditionsPlotterConfig):
        return SkillConditionsPlot(config)
    else:
        raise ValueError(f"Unknown SkillPlotterConfig type: {type(config)}")
