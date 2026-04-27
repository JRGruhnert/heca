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
        epochs: int = 100

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.heca = Heca.from_config(cfg.heca)

    def train(self):
        self.heca.train(self.cfg.epochs)

    def plot(self):
        for plotter in self.cfg.plotters:
            Plotter.from_config(plotter).init()
            plotter.plot()

    def explain(self):
        obs, goal = self.heca.sample()
        episode_ended = False
        while not episode_ended:
            self.heca.explain()
