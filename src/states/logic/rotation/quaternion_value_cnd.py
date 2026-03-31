from dataclasses import dataclass

import torch

from src.states.logic.rotation.quaternion import Quaternion, QuaternionConfig
from src.states.logic.value_cnd import ValueCondition, ValueConfig


@dataclass
class QuaternionValueConditionConfig(ValueConfig):
    rotation: QuaternionConfig = QuaternionConfig()


class QuaternionValueCondition(ValueCondition):
    def __init__(
        self,
        config: QuaternionValueConditionConfig,
    ):
        self.rotation = Quaternion(config.rotation)

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the quaternion."""
        return self.rotation.normalize_quat(x)

    def make_input(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the quaternion."""
        return self.value(x)
