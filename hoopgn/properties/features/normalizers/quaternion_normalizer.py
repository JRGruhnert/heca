from dataclasses import dataclass

import torch

from hoopgn.properties.features.quaternion import Quaternion
from hoopgn.properties.features.normalizers.normalizer import (
    PropertyNormalizer,
    PropertyNormalizerConfig,
)


@dataclass(kw_only=True)
class QuaternionNormalizerConfig(PropertyNormalizerConfig):
    pass


class QuaternionNormalizer(PropertyNormalizer):
    def __init__(
        self,
        config: QuaternionNormalizerConfig,
    ):
        super().__init__(config)
        self.config = config
        self.rotation = Quaternion()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the quaternion."""
        return self.rotation.normalize_quat(x)
