from abc import ABC, abstractmethod
from dataclasses import dataclass
import random
from hoopgn.misc import logger
from hoopgn.environments import select_environment
from hoopgn.environments.environment import Environment
from hoopgn.evaluators import select_evaluator
from hoopgn.evaluators.evaluator import Evaluator
from hoopgn.observation.td_properties import TDProperties
from hoopgn.agents.agent import Agent
import math


@dataclass(kw_only=True)
class ExperimentConfig:
    environments: list[Environment.Config]
    evaluator: Evaluator.Config


class Experiment(ABC):
    def __init__(self, cfg: ExperimentConfig):
        # We sort based on Id for the baseline network to be consistent
        self.cfg = cfg
        self.evaluator = select_evaluator(cfg.evaluator)
        self.environments = Environment.build_registry(cfg.environments)

        self.current_step = 0
        self.max_allowed_steps = math.inf
        self.current = self.env.reset()
        self.goal = self.env.reset()

    def step(self, skill: Agent) -> tuple[TDProperties, float, bool, bool]:
        selected_skill = self.modify(skill)
        selected_skill.reset(self.goal)
        while (action := selected_skill.predict(self.current)) is not None:
            feedback = self.env.step(action)  # TODO: feedback unused currently
            self.current = self.env.get_observation()
        reward, done = self.evaluator.step(self.current, self.goal)
        self.current_step += 1
        terminal = True if self.current_step >= self.max_allowed_steps else done
        logger.info(
            f"Step {self.current_step}: Reward={reward}, Done={done}, Terminal={terminal}"
        )
        return self.current, reward, done, terminal

    def sample_task(self) -> tuple[TDProperties, TDProperties]:
        logger.info("Sampling new task...")
        self.current_step = 0
        self.current = self.env.reset()
        self.goal = self.env.reset()
        attempts = 0
        while not self.evaluator.check_sample(self.current, self.goal):
            attempts += 1
            if attempts % 5 == 0:
                self.current = self.env.reset()
            self.goal = self.env.reset()
        return self.current, self.goal

    @abstractmethod
    def modify(self, skill: Agent) -> Agent:
        """Modify the skill before taking a step in the experiment."""
        raise NotImplementedError("Modify method not implemented yet.")

    @abstractmethod
    def metadata(self) -> dict:
        """Return experiment metadata as a dictionary."""
        raise NotImplementedError("Metadata method not implemented yet.")

    def close(self):
        self.env.close()
