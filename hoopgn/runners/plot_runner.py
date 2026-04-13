from hoopgn.plots.helper.run_data import TrainingRunData
from hoopgn.plots.hoopgn_plots import select_multiple_training_plots
from hoopgn.plots.hoopgn_plots.hoopgn_plot import HoopGNPlotConfig
from hoopgn.runners.runner import HoopGNRunner, HoopGNRunnerConfig
from dataclasses import dataclass
from glob import glob
import re


@dataclass
class PlotRunnerConfig(HoopGNRunnerConfig):
    plots: list[HoopGNPlotConfig]
    networks: list[str]
    file_pattern: re.Pattern = re.compile(
        r"(?P<tag>\w+_\w+_\w+)_pe(?P<pe>[0-9.]+)_pr(?P<pr>[0-9.]+)"
    )
    tag_pattern: re.Pattern = re.compile(
        r"(?P<ident>\w+)_(?P<origin>\w+)_(?P<dest>\w+)"
    )


class PlotRunner(HoopGNRunner):
    def __init__(self, config: PlotRunnerConfig):
        super().__init__(config)
        self.config = config
        self.plots = select_multiple_training_plots(config.plots)
        self.runs: list[TrainingRunData] = []
        self.load_results()

    def run(self):
        for plot in self.plots:
            plot(self.runs)

    def load_results(self):
        for nt in self.config.networks:
            self.load_network_results(nt)

    def load_network_results(self, nt: str):
        for path in glob(f"results/{nt}//*", recursive=True):
            file_match = self.config.file_pattern.search(path)
            if file_match:
                tag_match = self.config.tag_pattern.search(file_match.group("tag"))
                if tag_match:
                    metadata = {
                        "nt": nt,
                        "mode": tag_match.group("ident"),
                        "pe": float(file_match.group("pe")),
                        "pr": float(file_match.group("pr")),
                        "origin": tag_match.group("origin"),
                        "dest": tag_match.group("dest"),
                    }
                    self.runs.append(TrainingRunData(path, metadata))
