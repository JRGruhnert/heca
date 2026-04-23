import torch
import numpy as np
from abc import abstractmethod
from dataclasses import dataclass
from functools import cached_property
from tensordict import TensorDict

from hoopgn.classes import ConfigurableClass
from hoopgn.entities.entity import Entity
from hoopgn.properties.property import Property
from hoopgn.observation.td_entity import TDEntity


class LeafPolicy(ConfigurableClass):

    @dataclass(kw_only=True)
    class Config(ConfigurableClass.Config):
        pass

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
