from abc import abstractmethod
from dataclasses import dataclass

from heca.runners.plotters.plotter import Plotter, PlotterConfig


@dataclass
class HecaPlotterConfig(PlotterConfig):
    pass


class HoopGNPlot(Plotter):
    def __init__(self, config: HecaPlotterConfig):
        super().__init__(config)
        self.config = config

    @abstractmethod
    def plot_content(self):
        raise NotImplementedError()
