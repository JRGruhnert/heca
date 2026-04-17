from abc import abstractmethod
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import torch

from hoopgn.base import ConfigurableClass
from hoopgn.environments.properties.features.conditions.condition import (
    PropertyCondition,
)
from hoopgn.environments.properties.property import PropertyConfig


class BranchPolicy(ConfigurableClass):
    @dataclass(kw_only=True)
    class Config(ConfigurableClass.Config):
        properties: list[PropertyConfig] = field(default_factory=list)

    def __init__(self, cfg: Config):
        self.cfg = cfg

    @abstractmethod
    def __call__(self, x) -> np.ndarray | None:
        raise NotImplementedError()

    @abstractmethod
    def reset(self, goal):
        raise NotImplementedError()

    @abstractmethod
    def load_precons(self) -> dict[str, PropertyCondition]:
        raise NotImplementedError()

    @abstractmethod
    def load_postcons(self) -> dict[str, PropertyCondition]:
        raise NotImplementedError()

    @abstractmethod
    def load_demo_precons(self) -> dict[str, torch.Tensor]:
        raise NotImplementedError()

    @abstractmethod
    def load_demo_postcons(self) -> dict[str, torch.Tensor]:
        raise NotImplementedError()

    @cached_property
    def parameter_precons(self) -> dict[str, PropertyCondition]:
        return self.load_precons()

    @cached_property
    def parameter_postcons(self) -> dict[str, PropertyCondition]:
        return self.load_postcons()

    @cached_property
    def demo_precons(self) -> dict[str, torch.Tensor]:
        return self.load_demo_precons()

    @cached_property
    def demo_postcons(self) -> dict[str, torch.Tensor]:
        return self.load_demo_postcons()
