from abc import ABC, abstractmethod
from dataclasses import dataclass
from src.storage import select_states
from collections.abc import Sequence
from src.objects.properties.property import PropertyConfig
from src.observation.observation import StateValueDict


@dataclass(kw_only=True)
class EvaluatorConfig:
    success_reward: float
    states_eval: Sequence[PropertyConfig]
    states_network: Sequence[PropertyConfig]


class Evaluator(ABC):
    def __init__(self, config: EvaluatorConfig):
        self.config = config
        self.states_eval = select_states(config.states_eval)
        self.states_network = select_states(config.states_network)

        self.percentage_done: float = 0.0

    def is_equal(
        self,
        current: StateValueDict,
        goal: StateValueDict,
    ) -> bool:
        """Generic method to check if states match target conditions."""
        finished = 0
        for state in self.states_eval:
            label = state.config.label
            if label in current.keys():
                if state.evaluate(current[label], goal[label]):
                    finished += 1
        self.percentage_done = finished / max(len(self.states_eval), 1)
        return finished == len(self.states_eval)

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
        for state in self.states_network:
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
