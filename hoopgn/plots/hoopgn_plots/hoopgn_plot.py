from abc import abstractmethod
from dataclasses import dataclass
import re

from hoopgn.plots.helper.run_data import TrainingRunData
from hoopgn.plots.plot import Plot, PlotConfig


@dataclass
class HoopGNPlotConfig(PlotConfig):
    pass


class HoopGNPlot(Plot):
    def __init__(self, config: HoopGNPlotConfig):
        super().__init__(config)
        self.config = config

    @abstractmethod
    def __call__(self, runs: list[TrainingRunData]):
        pass

    def get(
        self,
        nt: str,
        mode: str,
        origin: str,
        dest: str,
        pe: float,
        pr: float,
        runs: list[TrainingRunData],
    ) -> TrainingRunData:
        for run in runs:
            if (
                run.metadata.get("nt") == nt
                and run.metadata.get("mode") == mode
                and run.metadata.get("origin") == origin
                and run.metadata.get("dest") == dest
                and run.metadata.get("pe") == pe
                and run.metadata.get("pr") == pr
            ):
                return run
        else:
            raise ValueError(
                f"No run found for network={nt}, mode={mode}, origin={origin}, dest={dest}, pe={pe}, pr={pr}"
            )
