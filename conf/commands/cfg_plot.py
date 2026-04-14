from conf.entity_set import properties_to_entities
from hoopgn.plotters.hoopgn_plotters.hoopgn_plotter import HoopGNPlotterConfig
from hoopgn.plotters.hoopgn_plotters.sampling_plotter import SpawnAreaPlotterConfig
from hoopgn.plotters.plotter import PlotterConfig
from hoopgn.runners.plot_runner import PlotRunnerConfig


from conf.property_sets import get_property_set
from conf.skill_sets import get_skill_set


SKILLS = "blue"
PROPERTIES_EVAL = "blue"
PROPERTIES_NETWORK = "blue"
ENTITIES = "blue"


skills = get_skill_set(SKILLS)
properties_eval = get_property_set(PROPERTIES_EVAL)
properties_network = get_property_set(PROPERTIES_NETWORK)
properties = properties_network
entities = properties_to_entities(properties=properties)

plot1 = SpawnAreaPlotterConfig()
plotters: list[HoopGNPlotterConfig] = [plot1]

cfg = PlotRunnerConfig(
    skills=skills,
    entities=entities,
    properties=properties,
    plotters=plotters,
)
