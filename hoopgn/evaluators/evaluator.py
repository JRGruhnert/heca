from abc import abstractmethod
from dataclasses import dataclass

from tensordict import TensorDict
from hoopgn.agents.agent import AgentFeedback
from hoopgn.misc.classes import ConfigClass
from hoopgn.misc.td import TDScene


class Evaluator(ConfigClass):
    @dataclass(kw_only=True)
    class Config(ConfigClass.Config):
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

    @abstractmethod
    def get_feedback(self, x: TensorDict) -> AgentFeedback:
        raise NotImplementedError()
