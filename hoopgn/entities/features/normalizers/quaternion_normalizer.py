from dataclasses import dataclass

import torch

from hoopgn.entities.properties.quaternion import Quaternion, QuaternionConfig
from hoopgn.entities.properties.features.normalizers.normalizer import (
    PropertyNormalizer,
    PropertyNormalizerConfig,
)


@dataclass(kw_only=True)
class QuaternionNormalizerConfig(PropertyNormalizerConfig):
    rotation: QuaternionConfig = QuaternionConfig()


class QuaternionNormalizer(PropertyNormalizer):
    def __init__(
        self,
        config: QuaternionNormalizerConfig,
    ):
        self.rotation = Quaternion(config.rotation)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the quaternion."""
        return self.rotation.normalize_quat(x)
