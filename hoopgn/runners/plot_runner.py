from hoopgn.plotters.hoopgn_plotters import select_multiple_hoopgn_plotters
from hoopgn.plotters.hoopgn_plotters.hoopgn_plotter import HoopGNPlotConfig
from hoopgn.runners.runner import HoopGNRunner, HoopGNRunnerConfig
from dataclasses import dataclass


@dataclass
class PlotRunnerConfig(HoopGNRunnerConfig):
    plots: list[HoopGNPlotConfig]


class PlotRunner(HoopGNRunner):
    def __init__(self, config: PlotRunnerConfig):
        super().__init__(config)
        self.config = config
        self.plotters = select_multiple_hoopgn_plotters(config.plots)

    def run(self):
        for plotter in self.plotters:
            plotter.plot()
