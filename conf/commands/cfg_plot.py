from conf.properties import get_property_set
from conf.skills import get_skill_set
from hoopgn.plotters.hoopgn_plotters.hoopgn_plotter import HoopGNPlotterConfig
from hoopgn.plotters.hoopgn_plotters.sampling_plotter import SpawnAreaPlotterConfig
from hoopgn.runners.plot_runner import PlotRunnerConfig


SKILL_TAG = "blue"
PROPERTY_TAG = "blue"

skills = get_skill_set(SKILL_TAG)
properties = get_property_set(PROPERTY_TAG)

plot1 = SpawnAreaPlotterConfig()
plotters: list[HoopGNPlotterConfig] = [plot1]

cfg = PlotRunnerConfig(
    skills=skills,
    properties=properties,
    plotters=plotters,
)
