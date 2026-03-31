from dataclasses import dataclass

import torch

from src.states.logic.quaternion import Quaternion, QuaternionConfig
from src.states.logic.value_handler.normalizers.normalizer import (
    Normalizer,
    NormalizerConfig,
)


@dataclass
class QuaternionNormalizerConfig(NormalizerConfig):
    rotation: QuaternionConfig = QuaternionConfig()


class RotationNormalizer(Normalizer):
    def __init__(
        self,
        config: QuaternionNormalizerConfig,
    ):
        self.rotation = Quaternion(config.rotation)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the quaternion."""
        return self.rotation.normalize_quat(x)
