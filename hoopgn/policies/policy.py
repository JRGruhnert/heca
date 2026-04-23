from typing import Any

import torch
from abc import abstractmethod
from dataclasses import dataclass
from functools import cached_property

from hoopgn.classes import ConfigurableClass
from hoopgn.entities.entity import Entity
from hoopgn.properties.property import Property
from hoopgn.observation.td_entity import TDEntity


class Policy(ConfigurableClass):

    @dataclass(kw_only=True)
    class Config(ConfigurableClass.Config):
        pass

    def __init__(self, cfg: Config):
        self.cfg = cfg

    @abstractmethod
    def __call__(self, x: TDEntity, y: TDEntity) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def from_disk(self, path: str):
        raise NotImplementedError()

    @cached_property
    @abstractmethod
    def ppre(self) -> dict[Property.Signature, torch.Tensor]:
        raise NotImplementedError()

    @cached_property
    @abstractmethod
    def ppost(self) -> dict[Property.Signature, torch.Tensor]:
        raise NotImplementedError()

    @cached_property
    @abstractmethod
    def epre(self) -> dict[Entity.Signature, TDEntity]:
        raise NotImplementedError()

    @cached_property
    @abstractmethod
    def epost(self) -> dict[Entity.Signature, TDEntity]:
        raise NotImplementedError()
