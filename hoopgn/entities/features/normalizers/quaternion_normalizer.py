from dataclasses import dataclass

import torch

from hoopgn.properties.rotations.quaternion import Quaternion
from hoopgn.properties.features.normalizers.normalizer import (
    PropertyNormalizer,
)


class QuaternionNormalizer(PropertyNormalizer):
    @dataclass(kw_only=True)
    class Config(PropertyNormalizer.Config):
        pass

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.rotation = Quaternion()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the quaternion."""
        return self.rotation.normalize_quat(x)
