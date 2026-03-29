from dataclasses import dataclass

import torch

from src.factory import select_distance_condition, select_state_preprocessor
from src.states.logic.distance import DistanceConfig
from src.states.logic.state_preprocessor import StatePreprocessorConfig


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

    def set(self, value: torch.Tensor | list[float] | float | int):
        self.value = self.preprocessor(value)

    def __call__(self, x: torch.Tensor) -> float:
        return self.distance(x, self.value)
