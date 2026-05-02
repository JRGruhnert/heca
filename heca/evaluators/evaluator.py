from abc import abstractmethod
from dataclasses import dataclass

from tensordict import TensorDict
from heca.agents.agent import AgentFeedback
from heca.classes.config import Configurable


class Evaluator(Configurable):
    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        success_reward: float

    @abstractmethod
    def reset(self, x: TensorDict, y: TensorDict):
        raise NotImplementedError()

    @abstractmethod
    def is_sample(self, x: TensorDict, y: TensorDict) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def step(self, x: TensorDict, feedback: AgentFeedback) -> tuple[float, bool]:
        raise NotImplementedError()
