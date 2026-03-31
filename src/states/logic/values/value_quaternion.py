from dataclasses import dataclass

import torch

from src.states.logic.quaternion import Quaternion, QuaternionConfig
from src.states.logic.values.value import Value, ValueHandlerConfig


@dataclass
class QuaternionValueConfig(ValueHandlerConfig):
    rotation: QuaternionConfig = QuaternionConfig()


class QuaternionValue(Value):
    def __init__(
        self,
        config: QuaternionValueConfig,
    ):
        self.rotation = Quaternion(config.rotation)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the quaternion."""
        return self.rotation.normalize_quat(x)

    def make_input(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the quaternion."""
        return self.__call__(x)
