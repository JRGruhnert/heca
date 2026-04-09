from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from src.objects.properties.condition import Condition, ConditionConfig


@dataclass
class NodeOperatorConfig:
    conditions: dict[str, ConditionConfig]


class NodeOperator(ABC):
    def __init__(self, config: NodeOperatorConfig):
        self.config = config

    @abstractmethod
    def __call__(self, start, goal) -> np.ndarray | None:
        "TODO: implement"
        raise NotImplementedError()

    @abstractmethod
    def reset(self, goal):
        "TODO: implement"
        raise NotImplementedError()

    @abstractmethod
    def load_parameter_precons(self) -> dict[str, Condition]:
        "TODO: implement"
        raise NotImplementedError()

    @abstractmethod
    def load_parameter_postcons(self) -> dict[str, Condition]:
        "TODO: implement"
        raise NotImplementedError()
