from abc import abstractmethod
from enum import Enum
from functools import cached_property
from hoopgn.base import RegisterableClass
from hoopgn.environments.entities.entity import Entity
from hoopgn.environments.properties.property import Property
from hoopgn.observation.td_scene import TDScene


class StepFeedback(Enum):
    OKAY = 0
    ERROR = 1


class Environment(RegisterableClass):

    @cached_property
    @abstractmethod
    def entities(self) -> list[Entity]:
        raise NotImplementedError()

    @cached_property
    @abstractmethod
    def properties(self) -> list[Property]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def observation(self) -> TDScene:
        raise NotImplementedError()

    @abstractmethod
    def sample(self) -> TDScene:
        raise NotImplementedError()

    @abstractmethod
    def step(self, action) -> StepFeedback:
        raise NotImplementedError()

    @abstractmethod
    def render(self):
        raise NotImplementedError()

    @abstractmethod
    def close(self):
        raise NotImplementedError()
