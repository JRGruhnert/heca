from conf.properties import get_property_set
from conf.skills import get_skill_set
from hoopgn.plotters.skill_plotters.condition_plotter import (
    SkillConditionsPlotterConfig,
)
from hoopgn.plotters.skill_plotters.skill_plotter import SkillPlotterConfig
from hoopgn.runners.skill.plot_skill_runner import SkillPlotRunnerConfig


plot1 = SkillConditionsPlotterConfig()

SKILL_TAG = "blue"
PROPERTY_TAG = "blue"

checkpoint_path = "checkpoints/explain/blue/best.pt"
skills = get_skill_set(SKILL_TAG)

properties = get_property_set(PROPERTY_TAG)

plotters: list[SkillPlotterConfig] = [plot1]  # TODO: add plots

cfg = SkillPlotRunnerConfig(
    skills=skills,
    properties=properties,
    plotters=plotters,
)
