from dataclasses import dataclass

import torch

from src.factory import select_distance, select_state_preprocessor
from src.states.addons.state_preprocessor import StatePreprocessorConfig
from src.states.rulers.ruler import RulerConfig


@dataclass
class ConditionConfig:
    ruler: RulerConfig
    preprocessor: StatePreprocessorConfig
    value: list[float] | float | int | None = None


class Condition:
    def __init__(self, config: ConditionConfig):
        self.config = config
        self.ruler = select_distance(config.ruler)
        self.preprocessor = select_state_preprocessor(config.preprocessor)
        self.value = self.preprocessor(config.value)

    def __call__(self, x: torch.Tensor, y: torch.Tensor | None = None) -> float:
        if y is not None:
            return self.ruler(x, y)
        return self.ruler(x, self.value)

    @classmethod
    def from_demos(cls, value: tuple, config: ConditionConfig) -> "Condition | None":
        instance = cls(config)
        start, end, reversed, selected_by_tapas = value
        instance.value = instance.preprocessor.hoopgnv1(
            start, end, reversed, selected_by_tapas
        )

        if instance.value is None:
            return None

        return instance
