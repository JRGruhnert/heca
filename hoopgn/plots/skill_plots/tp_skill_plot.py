from dataclasses import dataclass
import glob
import re

from hoopgn.plots.helper.run_data import TrainingRunData
from hoopgn.skills.skill import SkillConfig


@dataclass
class TrainingsPlotConfig:
    networks: list[str]
    file_pattern: re.Pattern
    tag_pattern: re.Pattern


class TrainingsPlot:
    def __init__(self, config: TrainingsPlotConfig):
        self.config = config
        self.runs: list[TrainingRunData] = []
        self.load_results()

    def plot(self, skill: SkillConfig):
        pass

    def load_results(self):
        for nt in self.config.networks:
            self.load_network_results(nt)

    def load_network_results(self, nt: str):
        for path in glob.glob(f"results/{nt}//*", recursive=True):
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

    def get(
        self,
        nt: str,
        mode: str,
        origin: str,
        dest: str,
        pe: float,
        pr: float,
    ) -> TrainingRunData:
        for run in self.runs:
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
