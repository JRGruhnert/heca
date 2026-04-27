from dataclasses import dataclass

from heca.agents.hecas.heca import Heca
from heca.misc.classes import ConfigClass

from heca.runners.plotters.hoopgn_plotters.hoopgn_plotter import HecaPlotterConfig
from heca.runners.plotters.plotter import Plotter


class HecaRunner(ConfigClass):
    @dataclass(kw_only=True)
    class Config(ConfigClass.Config):
        heca: Heca.Config
        plotters: list[HecaPlotterConfig]
        episodes: int = 1000

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def train(self):
        assert self.cfg.heca, "No Heca config provided"
        assert self.cfg.episodes > 0, "Number of episodes must be greater than 0"
        self.heca = Heca.search(self.cfg.heca.query)
        self.heca.train(self.cfg.episodes)

    def plot(self):
        assert self.cfg.plotters, "No plotters specified in config"
        for plotter in self.cfg.plotters:
            plotter = Plotter.from_config(plotter)
            plotter.plot()

    def explain(self):
        assert self.cfg.heca, "No Heca config provided"
        self.heca = Heca.search(self.cfg.heca.query)
        self.heca.explain()
