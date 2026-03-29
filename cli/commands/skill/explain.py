from dataclasses import dataclass
from src.modules.storage import Storage, StorageConfig
from src.plotting.object_point import ObjectLocationPoint
from src.plotting.plots.skill.tp import ObjectConditionsPlot
from src.skills.tree.leafs.leaf import Leaf


@dataclass
class SkillExplainManagerConfig:
    storage: StorageConfig


class SkillExplainScript:
    def __init__(self, config: SkillExplainManagerConfig):
        self.config = config
        self.storage = Storage(config.storage)
        self.plot = ObjectConditionsPlot()
        self.object_labels: list[str] = ["ee", "red_block", "blue_block", "pink_block"]

    def run(self):
        for skill in self.storage.skills:
            self.make_explanation(skill)

    def make_point(self, con: dict, label: str) -> ObjectLocationPoint:
        return ObjectLocationPoint(
            x=con[f"{label}_position"][0].item(),
            y=con[f"{label}_position"][1].item(),
            z=con[f"{label}_position"][2].item(),
            rotation=con[f"{label}_rotation"].item(),
            state=int(con[f"{label}_state"].item()),
        )

    def make_explanation(self, skill: Leaf):
        """Returns an explanation for the given observation, goal and skill."""
        pre = skill.demo_precons
        post = skill.demo_postcons
        for o in self.object_labels:
            pre_con = self.make_point(pre, o)
            post_con = self.make_point(post, o)
            self.plot.set_precon(pre_con)
            self.plot.set_postcon(post_con)

        self.plot.create(
            title=f"{skill.config.label} - Taskparameters.",
            show=True,
            save=False,
            path=f"plots/{skill.config.label}_tps.png",
        )


def entry_point(config: SkillExplainManagerConfig):
    script = SkillExplainScript(config)
    script.run()
