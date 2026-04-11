from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from hoopgn.observation.converters.converter import Converter, ConverterConfig
from hoopgn.observation.observation import StateValueDict


class StepFeedback(Enum):
    OKAY = 0
    ERROR = 1


@dataclass(kw_only=True)
class EnvironmentConfig:
    converters: list[ConverterConfig] = field(default_factory=list)


class Environment(ABC):
    def __init__(self, config: EnvironmentConfig):
        self.config = config
        self.converters = [Converter(config=config) for config in config.converters]

    def reset(self) -> StateValueDict:
        self._reset()
        return self.get_observation()

    @abstractmethod
    def _reset(self) -> StateValueDict:
        raise NotImplementedError("Internal reset method not implemented yet.")

    @abstractmethod
    def step(self, action) -> StepFeedback:
        raise NotImplementedError("Step method not implemented yet.")

    @abstractmethod
    def render(self):
        raise NotImplementedError("Render method not implemented yet.")

    @abstractmethod
    def get_observation(self) -> StateValueDict:
        raise NotImplementedError("Get observation method not implemented yet.")

    @abstractmethod
    def close(self):
        raise NotImplementedError("Close method not implemented yet.")
