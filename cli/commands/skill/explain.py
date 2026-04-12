from dataclasses import dataclass

import torch
from hoopgn.entities.properties.features.conditions.condition import PropertyCondition
from hoopgn.plotting.object_point import ObjectLocationPoint
from hoopgn.plotting.plots.skill.tp import ObjectConditionsPlot
from hoopgn.skills.skill import Skill, SkillConfig


@dataclass
class SkillExplainManagerConfig:
    skill: SkillConfig
    plot_demos: bool = True
    plot_task_params: bool = True


class SkillExplainScript:
    def __init__(self, config: SkillExplainManagerConfig):
        self.config = config
        self.skill = Skill(config.skill)
        self.plot = ObjectConditionsPlot()
        self.entity_labels: list[str] = [
            "ee",
            "block_red",
            "block_blue",
            "block_pink",
            "slide",
            "drawer",
            "button",
            "led",
            "lightbulb",
        ]

    def run(self):
        self.make_explanation(self.skill)

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

    def make_explanation(self, skill: Skill):

        for label in self.entity_labels:
            # Demo points
            pre_point = self.make_demo_point(label, self.skill.demo_precons)
            post_point = self.make_demo_point(label, self.skill.demo_postcons)
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


def entry_point(config: SkillExplainManagerConfig):
    script = SkillExplainScript(config)
    script.run()
