from dataclasses import dataclass
import glob
import re

from hoopgn.data import TrainingRunData
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

    def run(self, skill: Skill):
        for entity in self.entities:
            label = entity.config.label
            # Demo points
            pre_point = self.make_demo_point(label, skill.demo_precons)
            post_point = self.make_demo_point(label, skill.demo_postcons)
            self.plot.set_precon(pre_point)
            self.plot.set_postcon(post_point)
            # TP points
            tp_pre = self.make_tp_point(label, skill.precons)
            tp_post = self.make_tp_point(label, skill.postcons)
            self.plot.set_precon_tp(tp_pre)
            self.plot.set_postcon_tp(tp_post)

        self.plot.create(
            title=f"{skill.config.label} - Taskparameters.",
            show=True,
            save=False,
            path=f"plots/{skill.config.label}_tps.png",
        )

    def make_demo_point(
        self, label: str, value: dict[str, torch.Tensor]
    ) -> ObjectLocationPoint:
        pos = value[f"{label}_position"].flatten()
        rot = value[f"{label}_rotation"].flatten()
        state = value[f"{label}_scalar"].flatten()
        return ObjectLocationPoint(
            x=pos[0].item(),
            y=pos[1].item(),
            z=pos[2].item(),
            rotation=rot[0].item(),
            state=int(state[0].item()),
            label=label,
        )

    def make_tp_point(
        self, label: str, conditions: dict[str, PropertyCondition]
    ) -> ObjectLocationPoint:
        pos = conditions[f"{label}_position"].value.flatten()
        rot = conditions[f"{label}_rotation"].value.flatten()
        state = conditions[f"{label}_scalar"].value.flatten()
        return ObjectLocationPoint(
            x=pos[0].item(),
            y=pos[1].item(),
            z=pos[2].item(),
            rotation=rot[0].item(),
            state=int(state[0].item()),
            label=label,
        )
