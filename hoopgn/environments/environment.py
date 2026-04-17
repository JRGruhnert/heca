from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from hoopgn.base import ConfigurableClass
from hoopgn.environments.entities.entity import Entity
from hoopgn.observation.converters import select_observation_converter
from hoopgn.observation.converters.converter import ConverterConfig
from hoopgn.observation.converters.tapas_converter import TapasConverterConfig
from hoopgn.observation.td_scene import TDScene


class StepFeedback(Enum):
    OKAY = 0
    ERROR = 1


class Environment(ConfigurableClass):
    @dataclass(kw_only=True)
    class Config(ConfigurableClass.Config):
        label: str
        converters: list[ConverterConfig] = field(
            default_factory=lambda: [TapasConverterConfig()]
        )

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.converters = [select_observation_converter(cfg) for cfg in cfg.converters]

    def reset(self) -> TDScene:
        self._reset()
        return self.get_observation()

    @property
    def entities(self) -> list[Entity]:
        return self.get_observation().entities

    @property
    def properties(self) -> list[str]:
        return self.get_observation().properties

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
    def get_observation(self) -> TDScene:
        raise NotImplementedError("Get observation method not implemented yet.")

    @abstractmethod
    def close(self):
        raise NotImplementedError("Close method not implemented yet.")
