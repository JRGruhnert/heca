from dataclasses import dataclass
from src.modules.storage import Storage, StorageConfig
from dataclasses import dataclass
from src.modules.storage import Storage, StorageConfig
from src.plotting.object_point import ObjectLocationPoint
from src.plotting.plots.object_skill import ObjectConditionsPlot
from src.skills.skill import Skill


@dataclass
class SkillExplainManagerConfig:
    storage: StorageConfig


class SkillExplainScript:
    def __init__(self, config: SkillExplainManagerConfig):
        self.config = config
        self.storage = Storage(config.storage)
        self.plot = ObjectConditionsPlot()

    def run(self):
        for skill in self.storage.skills:
            self.make_explanation(skill)

    def make_explanation(self, skill: Skill):
        """Returns an explanation for the given observation, goal and skill."""
        for demo in skill.demo_precons:
            for pre in demo.items():
                # TODO:
                point = ObjectLocationPoint(
                    x=pre[1][0].item(),
                    y=pre[1][1].item(),
                    z=pre[1][2].item(),
                    rotation=pre[1][3].item(),
                    state=int(pre[1][3].item()),
                )
                self.plot.set_precon(point)

        self.plot.create(
            title=f"{skill.name} - Taskparameters.",
            show=True,
            save=False,
            path=f"plots/{skill.name}_tps.png",
        )


def entry_point(config: SkillExplainManagerConfig):
    script = SkillExplainScript(config)
    script.run()
