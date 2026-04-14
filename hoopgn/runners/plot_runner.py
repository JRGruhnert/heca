from hoopgn.plotters.hoopgn_plotters import select_multiple_hoopgn_plotters
from hoopgn.plotters.hoopgn_plotters.hoopgn_plotter import HoopGNPlotterConfig
from hoopgn.runners.runner import HoopGNRunner, HoopGNRunnerConfig
from dataclasses import dataclass


@dataclass
class PlotRunnerConfig(HoopGNRunnerConfig):
    plotters: list[HoopGNPlotterConfig]


class PlotRunner(HoopGNRunner):
    def __init__(self, config: PlotRunnerConfig):
        super().__init__(config)
        self.config = config
        self.plotters = select_multiple_hoopgn_plotters(config.plotters)

    def run(self):
        for plotter in self.plotters:
            plotter.plot()
