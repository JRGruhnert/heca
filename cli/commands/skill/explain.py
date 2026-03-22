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

from src.plotting.plots.positions import Positions3DPlot
from src.skills.skill import Skill


@dataclass
class SkillExplainManagerConfig:
    storage: StorageConfig


class SkillExplainScript:
    def __init__(self, config: SkillExplainManagerConfig):
        self.storage = Storage(config.storage)
        self.plot = Positions3DPlot()

    def run(self):
        for skill in self.storage.skills:
            self.make_explanation(skill)

    def make_explanation(self, skill: Skill):
        """Returns an explanation for the given observation, goal and skill."""
        pattern = re.compile(r"(.*)_(position)$")
        quick_map = {"blue": "b", "pink": "p", "red": "r"}

        for key, precon in skill.precons.items():
            m = pattern.match(key)
            if m:
                postcon = skill.postcons[key]
                assert postcon is not None, f"Post condition for {key} is missing"
                prefix = m.group(1)
                self.plot.print_object(
                    precon.tolist(),
                    skill.precons[f"{prefix}_rotation"].tolist(),
                    color=quick_map.get(prefix, "k"),
                )
                self.plot.print_object(
                    postcon.tolist(),
                    skill.postcons[f"{prefix}_rotation"].tolist(),
                    color=quick_map.get(prefix, "k"),
                )
                self.plot.connect_objects(
                    precon.tolist(),
                    postcon.tolist(),
                    color=quick_map.get(prefix, "k"),
                    alpha=0.5,
                )
        self.plot.create(show=True, save=False)


def entry_point(config: SkillExplainManagerConfig):
    script = SkillExplainScript(config)
    script.run()
