from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from hoopgn.base import ConfigurableClass, RegisterableClass
from hoopgn.environments.entities.entity import Entity
from hoopgn.observation.converters.converter import Converter
from hoopgn.observation.converters.tapas_converter import TapasConverterConfig
from hoopgn.observation.td_scene import TDScene


class StepFeedback(Enum):
    OKAY = 0
    ERROR = 1


class Environment(RegisterableClass):
    @dataclass(kw_only=True)
    class Signature(RegisterableClass.Signature):
        label: str

    @dataclass(kw_only=True)
    class Config(RegisterableClass.Config):
        pass

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def reset(self) -> TDScene:
        self._reset()
        return self.observation

    @cached_property
    @abstractmethod
    def entities(self) -> list[Entity]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def observation(self) -> TDScene:
        raise NotImplementedError()

    @abstractmethod
    def _reset(self) -> TDScene:
        raise NotImplementedError("Internal reset method not implemented yet.")

    @abstractmethod
    def step(self, action) -> StepFeedback:
        raise NotImplementedError("Step method not implemented yet.")

    @abstractmethod
    def render(self):
        raise NotImplementedError("Render method not implemented yet.")

    @abstractmethod
    def close(self):
        raise NotImplementedError("Close method not implemented yet.")
