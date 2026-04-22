from abc import abstractmethod
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from tensordict import TensorDict
import torch

from hoopgn.base import ConfigurableClass
from hoopgn.entities.entity import Entity
from hoopgn.entities.features.conditions.condition import EntityCondition
from hoopgn.observation.td_entity import TDEntity
from hoopgn.properties.features.conditions.condition import (
    PropertyCondition,
)
from hoopgn.properties.property import Property


class Policy(ConfigurableClass):
    @dataclass(kw_only=True)
    class Config(ConfigurableClass.Config):
        label: str

    def __init__(self, cfg: Config):
        self.cfg = cfg

    @abstractmethod
    def __call__(self, x: TensorDict, y: TensorDict) -> np.ndarray | None:
        raise NotImplementedError()

    @abstractmethod
    def from_disk(self, path: str):
        raise NotImplementedError()

    @cached_property
    def ppre(self) -> dict[Property.Signature, torch.Tensor]:
        raise NotImplementedError()

    @cached_property
    def ppost(self) -> dict[Property.Signature, torch.Tensor]:
        raise NotImplementedError()

    @cached_property
    def epre(self) -> dict[Entity.Signature, TDEntity]:
        raise NotImplementedError()

    @cached_property
    def epost(self) -> dict[Entity.Signature, TDEntity]:
        raise NotImplementedError()
