from conf.entity_set import properties_to_entities

from conf.property_sets import get_property_set
from conf.skill_sets import get_skill_set
from hoopgn.plotters.skill_plotters.cnd_plotter import SkillConditionsPlotterConfig
from hoopgn.plotters.skill_plotters.skill_plotter import SkillPlotterConfig
from hoopgn.runners.skill.plot_skill_runner import SkillPlotRunnerConfig


SKILLS = "blue"
PROPERTIES_EVAL = "blue"
PROPERTIES_NETWORK = "blue"
plot1 = SkillConditionsPlotterConfig()

skills = get_skill_set(SKILLS)
properties_eval = get_property_set(PROPERTIES_EVAL)
properties_network = get_property_set(PROPERTIES_NETWORK)
properties = properties_network
entities = properties_to_entities(properties=properties)

plotters: list[SkillPlotterConfig] = [plot1]  # TODO: add plots

cfg = SkillPlotRunnerConfig(
    skills=skills,
    entities=entities,
    properties=properties,
    plotters=plotters,
)
