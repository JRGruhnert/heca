from abc import abstractmethod
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import torch

from hoopgn.base import ConfigurableClass
from hoopgn.entities.features.conditions.condition import EntityCondition
from hoopgn.properties.features.conditions.condition import (
    PropertyCondition,
)


class Policy(ConfigurableClass):
    @dataclass(kw_only=True)
    class Config(ConfigurableClass.Config):
        pass

    def __init__(self, cfg: Config):
        self.cfg = cfg

    @abstractmethod
    def __call__(self, x) -> np.ndarray | None:
        raise NotImplementedError()

    @abstractmethod
    def reset(self, goal):
        raise NotImplementedError()

    @abstractmethod
    def load_precons(self) -> dict[str, EntityCondition | PropertyCondition]:
        raise NotImplementedError()

    @abstractmethod
    def load_postcons(self) -> dict[str, EntityCondition | PropertyCondition]:
        raise NotImplementedError()

    @cached_property
    def precons(self) -> dict[str, EntityCondition | PropertyCondition]:
        return self.load_precons()

    @cached_property
    def postcons(self) -> dict[str, EntityCondition | PropertyCondition]:
        return self.load_postcons()
