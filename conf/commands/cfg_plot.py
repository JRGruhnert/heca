from heca.agents.hecas.heca import Heca
from heca.agents.hecas.mps.red import RedMPHeca
from heca.runners.plotters.hecas.heca_plotter import HecaPlotterConfig
from heca.runners.plotters.hecas.heca_sampling import (
    SpawnAreaPlotterConfig,
)

from heca.runners.runner import HecaRunner

plot1 = SpawnAreaPlotterConfig()
plotters: list[HecaPlotterConfig] = [plot1]

red_mp = RedMPHeca.Query(
    label="red",
)

heca = RedMPHeca.search_config(query=RedMPHeca.Query(label="red"))
assert isinstance(heca, Heca.Config)
cfg = HecaRunner.Config(
    heca=heca,
    plotters=plotters,
)
