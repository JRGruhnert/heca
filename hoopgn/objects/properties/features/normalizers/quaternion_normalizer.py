from dataclasses import dataclass

import torch

from hoopgn.objects.properties.quaternion import Quaternion, QuaternionConfig
from hoopgn.objects.properties.features.normalizers.normalizer import (
    Normalizer,
    NormalizerConfig,
)


@dataclass(kw_only=True)
class QuaternionNormalizerConfig(NormalizerConfig):
    rotation: QuaternionConfig = QuaternionConfig()


class QuaternionNormalizer(Normalizer):
    def __init__(
        self,
        config: QuaternionNormalizerConfig,
    ):
        self.rotation = Quaternion(config.rotation)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the quaternion."""
        return self.rotation.normalize_quat(x)
