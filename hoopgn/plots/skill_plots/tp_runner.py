from dataclasses import dataclass

import torch
from hoopgn.properties.features.conditions.condition import PropertyCondition
from hoopgn.plots.helper.object_point import ObjectLocationPoint
from hoopgn.plots.helper.skill.tp import ObjectConditionsPlot
from hoopgn.runners.skill.skill_runner import SkillRunner, SkillRunnerConfig
from hoopgn.skills.skill import Skill


@dataclass
class SkillPlotterConfig(SkillRunnerConfig):
    plot_demos: bool = True
    plot_task_params: bool = True


class SkillPlotter(SkillRunner):
    def __init__(self, config: SkillPlotterConfig):
        super().__init__(config)
        self.config = config
        self.skills = [Skill(skill_config) for skill_config in config.skills]
        self.plot = ObjectConditionsPlot()

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
