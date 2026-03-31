from dataclasses import dataclass

import torch

from src.factory import select_distance_condition, select_state_preprocessor
from src.states.logic.distance import DistanceConfig
from src.states.logic.addons.state_preprocessor import StatePreprocessorConfig


@dataclass
class ConditionConfig:
    distance: DistanceConfig
    preprocessor: StatePreprocessorConfig
    value: list[float] | float | int


class Condition:
    def __init__(self, config: ConditionConfig):
        self.config = config
        self.distance = select_distance_condition(config.distance)
        self.preprocessor = select_state_preprocessor(config.preprocessor)
        self.value = self.preprocessor(config.value)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> float:
        return self.distance(x, y)

    @classmethod
    def from_demos(cls, value: tuple, config: ConditionConfig) -> "Condition":
        instance = cls(config)
        instance.value = instance.preprocessor(value)
        return instance
