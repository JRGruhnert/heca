from dataclasses import dataclass
import re
from src.modules.storage import Storage, StorageConfig
from collections import defaultdict
from dataclasses import dataclass
from typing import Any
from src.modules.logger import LoggerConfig, Logger
from src.modules.buffer import BufferConfig, Buffer
from src.modules.evaluators.evaluator import EvaluatorConfig
from src.modules.storage import Storage, StorageConfig
from src.environments.environment import EnvironmentConfig
from src.agents.ppo import PPOAgentConfig
from src.experiments.experiment import ExperimentConfig
from wandb.wandb_run import Run
from src.factory import (
    select_agent,
    select_environment,
    select_experiment,
    select_evaluator,
)
import numpy as np

from src.plotting.plots.positions_skill import PositionsSkill3DPlot
from src.skills.skill import Skill
from src.variables import SET_BLUE, SET_PINK, SET_RED, SET_SLIDE


@dataclass
class SkillExplainManagerConfig:
    storage: StorageConfig


class SkillExplainScript:
    def __init__(self, config: SkillExplainManagerConfig):
        self.config = config
        self.storage = Storage(config.storage)
        self.plot = PositionsSkill3DPlot()

        self.relevant_objects = {
            SET_RED: "block_red",
            SET_BLUE: "block_blue",
            SET_PINK: "block_pink",
            SET_SLIDE: "block_slide",
        }
        self.object_pattern = re.compile(r"(red|blue|pink|slide)")
        # print(f"Checkpoint path: {self.config.agent.network.checkpoint_path}")
        match = self.object_pattern.search(
            self.config.agent.network.checkpoint_path or ""
        )
        if match:
            object_name = match.group(1)  # 'red', 'blue', 'pink', or 'slide'
        self.trained_object = self.relevant_objects[object_name]
        self.current_object = self.relevant_objects[self.config.storage.used_states]

    def run(self):
        for skill in self.storage.skills:
            self.make_explanation(skill)

    def make_explanation(self, skill: Skill):
        """Returns an explanation for the given observation, goal and skill."""
        pattern = re.compile(r"(.*)_(position)$")

        for key, precon in skill.precons.items():
            print(
                f"Processing precondition {key} for with value {precon} skill {skill.name}"
            )
            m = pattern.match(key)
            if m:
                postcon = skill.postcons[key]
                assert postcon is not None, f"Post condition for {key} is missing"
                prefix = m.group(1)
                self.plot.set_object(
                    pos={"current": precon.tolist(), "goal": postcon.tolist()},
                    quat={
                        "current": skill.precons[f"{prefix}_rotation"].tolist(),
                        "goal": skill.postcons[f"{prefix}_rotation"].tolist(),
                    },
                    different=not np.allclose(precon, postcon),
                    solved=skill.is_solved(precon, postcon),
                )

                self.plot.show_edges(
                    alpha=0.5,
                )
        self.plot.create(
            title=f"Skill: {skill.name} - Taskparameters.",
            show=True,
            save=False,
            path=f"{self.storage.agent_saving_path(self.config.agent.network.name)}/plots/{self.trained_object}_s_{self.current_object}.png",
        )


def entry_point(config: SkillExplainManagerConfig):
    script = SkillExplainScript(config)
    script.run()
