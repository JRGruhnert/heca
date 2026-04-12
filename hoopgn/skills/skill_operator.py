from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import torch

from hoopgn.properties.features.conditions.condition import (
    PropertyCondition,
    PropertyConditionConfig,
)


@dataclass(kw_only=True)
class SkillOperatorConfig:
    conditions: dict[str, PropertyConditionConfig]


class SkillOperator(ABC):
    def __init__(self, config: SkillOperatorConfig):
        self.config = config

    @abstractmethod
    def __call__(self, x) -> np.ndarray | None:
        "TODO: implement"
        raise NotImplementedError()

    @abstractmethod
    def reset(self, goal):
        "TODO: implement"
        raise NotImplementedError()

    @abstractmethod
    def load_parameter_precons(self) -> dict[str, PropertyCondition]:
        "TODO: implement"
        raise NotImplementedError()

    @abstractmethod
    def load_parameter_postcons(self) -> dict[str, PropertyCondition]:
        "TODO: implement"
        raise NotImplementedError()

    @abstractmethod
    def load_demo_precons(self) -> dict[str, torch.Tensor]:
        "TODO: implement"
        raise NotImplementedError()

    @abstractmethod
    def load_demo_postcons(self) -> dict[str, torch.Tensor]:
        "TODO: implement"
        raise NotImplementedError()

    @cached_property
    def parameter_precons(self) -> dict[str, PropertyCondition]:
        return self.load_parameter_precons()

    @cached_property
    def parameter_postcons(self) -> dict[str, PropertyCondition]:
        return self.load_parameter_postcons()

    @cached_property
    def demo_precons(self) -> dict[str, torch.Tensor]:
        return self.load_demo_precons()

    @cached_property
    def demo_postcons(self) -> dict[str, torch.Tensor]:
        return self.load_demo_postcons()
