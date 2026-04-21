from abc import abstractmethod
from dataclasses import dataclass
from functools import cached_property
from tensordict import TensorDict

from hoopgn.base import ConfigurableClass
from hoopgn.entities.entity import Entity
from hoopgn.properties.property import Property


class Converter(ConfigurableClass):
    @dataclass(kw_only=True)
    class Config(ConfigurableClass.Config):
        label: str

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def __call__(self, observation) -> TensorDict:
        raise NotImplementedError(
            "ObservationConverter __call__ method not implemented yet."
        )

    @cached_property
    @abstractmethod
    def entities(self) -> list[Entity]:
        raise NotImplementedError()

    @cached_property
    @abstractmethod
    def properties(self) -> list[Property]:
        raise NotImplementedError()
