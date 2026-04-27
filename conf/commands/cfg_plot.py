from conf.properties import get_property_set
from conf.agents import get_skill_set
from heca.runners.plotters.hoopgn_plotters.hoopgn_plotter import HecaPlotterConfig
from heca.runners.plotters.hoopgn_plotters.sampling_plotter import (
    SpawnAreaPlotterConfig,
)
from heca.runners.plotter import PlotRunnerConfig


SKILL_TAG = "blue"
PROPERTY_TAG = "blue"

skills = get_skill_set(SKILL_TAG)
properties = get_property_set(PROPERTY_TAG)

plot1 = SpawnAreaPlotterConfig()
plotters: list[HecaPlotterConfig] = [plot1]

cfg = PlotRunnerConfig(
    skills=skills,
    properties=properties,
    plotters=plotters,
)
