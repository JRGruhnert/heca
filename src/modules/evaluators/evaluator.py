from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.modules.storage import Storage
from src.observation.observation import StateValueDict


@dataclass
class EvaluatorConfig:
    success_reward: float


class Evaluator(ABC):
    def __init__(
        self,
        storage: Storage,
    ):
        self.storage = storage
        self.percentage_done: float = 0.0

    def is_equal(
        self,
        current: StateValueDict,
        goal: StateValueDict,
    ) -> bool:
        """Generic method to check if states match target conditions."""
        finished = 0
        for label, state in self.storage.states_dict_eval.items():
            if label in current.keys():
                if state.evaluate(current[label], goal[label]):
                    finished += 1
        self.percentage_done = finished / max(len(self.storage.states_eval), 1)
        return finished == len(self.storage.states_eval)

    def evaluate_sample(
        self,
        current: StateValueDict,
        goal: StateValueDict,
    ) -> bool:
        done = self.is_equal(current, goal)
        valid = self.is_valid(current, goal)
        return not done and valid

    def is_valid(
        self,
        current: StateValueDict,
        goal: StateValueDict,
    ) -> bool:
        """Special method to check wether the sampled states are buggy or not."""
        for state in self.storage.states_network:
            if not state.validate(
                current[state.config.label],
                goal[state.config.label],
            ):
                return False
        return True

    @abstractmethod
    def step(
        self,
        current: StateValueDict,
        goal: StateValueDict,
    ) -> tuple[float, bool]:
        "Returns the step reward and wether the step is a terminal step, cause some ending condition was met."
        raise NotImplementedError()
