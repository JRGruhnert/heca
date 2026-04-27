from abc import abstractmethod
from dataclasses import dataclass

from heca.plotters.plotter import Plotter, PlotterConfig


@dataclass
class HoopGNPlotterConfig(PlotterConfig):
    pass


class HoopGNPlot(Plotter):
    def __init__(self, config: HoopGNPlotterConfig):
        super().__init__(config)
        self.config = config

    @abstractmethod
    def plot_content(self):
        raise NotImplementedError()
