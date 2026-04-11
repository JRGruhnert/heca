from abc import ABC, abstractmethod
from dataclasses import dataclass
from hoopgn.environments import select_environment
from hoopgn.environments.environment import EnvironmentConfig
from hoopgn.evaluators import select_evaluator
from hoopgn.evaluators.evaluator import EvaluatorConfig
from hoopgn.observation.observation import StateValueDict
from hoopgn.skills.skill import Skill
import math


@dataclass(kw_only=True)
class ExperimentConfig:
    environment: EnvironmentConfig
    evaluator: EvaluatorConfig


class Experiment(ABC):
    def __init__(self, config: ExperimentConfig):
        # We sort based on Id for the baseline network to be consistent
        self.config = config
        self.evaluator = select_evaluator(config.evaluator)
        self.env = select_environment(config.environment)

        self.current_step = 0
        self.max_allowed_steps = math.inf
        self.current = self.env.reset()
        self.goal = self.env.reset()

    def step(self, skill: Skill) -> tuple[StateValueDict, float, bool, bool]:
        selected_skill = self.modify(skill)
        selected_skill.reset(self.goal)
        while (action := selected_skill.predict(self.current)) is not None:
            feedback = self.env.step(action)  # TODO: feedback unused currently
            self.current = self.env.get_observation()
        reward, done = self.evaluator.step(self.current, self.goal)
        self.current_step += 1
        terminal = True if self.current_step >= self.max_allowed_steps else done
        return self.current, reward, done, terminal

    def sample_task(self) -> tuple[StateValueDict, StateValueDict]:
        self.current_step = 0
        self.current = self.env.reset()
        self.goal = self.env.reset()
        while not self.evaluator.evaluate_sample(self.current, self.goal):
            self.goal = self.env.reset()
        return self.current, self.goal

    @abstractmethod
    def modify(self, skill: Skill) -> Skill:
        """Modify the skill before taking a step in the experiment."""
        raise NotImplementedError("Modify method not implemented yet.")

    @abstractmethod
    def metadata(self) -> dict:
        """Return experiment metadata as a dictionary."""
        raise NotImplementedError("Metadata method not implemented yet.")

    def close(self):
        self.env.close()
