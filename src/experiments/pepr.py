from dataclasses import dataclass
import math
import random

from loguru import logger
from src.modules.storage import Storage
from src.experiments.experiment import Experiment, ExperimentConfig
from src.environments.environment import Environment
from src.observation.observation import StateValueDict
from src.skills.tree.leafs.leaf import Leaf
from src.skills.tree.leafs.leaf_ignore import IgnoreLeaf


@dataclass
class PePrConfig(ExperimentConfig):
    p_empty: float
    p_rand: float
    min_steps: int


class PePrExperiment(Experiment):
    """Simple Wrapper for centralized data loading and initialisation."""

    def __init__(self, config: PePrConfig, env: Environment, storage: Storage):
        # We sort based on Id for the baseline network to be consistent
        super().__init__(config, env, storage)
        self.config = config

        print(storage.config.skills)
        # NOTE: This is for my skills setup
        self.max_episode_length = math.ceil(
            self.config.min_steps
            + self.config.min_steps * self.config.p_empty
            + self.config.min_steps * self.config.p_rand
        )

        self.current_step = 0
        self.current: StateValueDict | None = None
        logger.info(
            f"Number of skills: {self.config.min_steps}, max episode length: {self.max_episode_length}"
        )

    def sample_task(self) -> tuple[StateValueDict, StateValueDict]:
        self.current_step = 0
        self.current, goal = self.env.sample_task()
        return self.current, goal

    def step(self, skill: Leaf) -> tuple[StateValueDict, float, bool, bool]:
        self.current_step += 1
        sample = random.random()
        if sample < self.config.p_empty:
            logger.info("Taking Empty Step")
            overwrite_skill = IgnoreLeaf()
        elif sample < self.config.p_empty + self.config.p_rand:
            logger.info("Taking Random Step")
            overwrite_skill = random.choice(self.storage.skills)
        else:  # The rest
            overwrite_skill = skill

        self.current, reward, done = self.env.step(overwrite_skill)
        terminal = True if self.current_step >= self.max_episode_length else done
        return self.current, reward, done, terminal

    def metadata(self) -> dict:
        return {
            "p_empty": self.config.p_empty,
            "p_rand": self.config.p_rand,
            "max_episode_length": self.max_episode_length,
        }
