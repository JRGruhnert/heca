from dataclasses import dataclass

import torch

from hoopgn.environments.properties.features.normalizers.normalizer import (
    PropertyNormalizer,
)


class IgnoreNormalizer(PropertyNormalizer):
    @dataclass(kw_only=True)
    class Config(PropertyNormalizer.Config):
        pass

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x
