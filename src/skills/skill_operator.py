from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from src.objects.properties.property_condition import (
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
