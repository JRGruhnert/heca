from dataclasses import dataclass

import torch

from hoopgn.entities.properties.features.normalizers.normalizer import (
    PropertyNormalizer,
    PropertyNormalizerConfig,
)


@dataclass(kw_only=True)
class IgnoreNormalizerConfig(PropertyNormalizerConfig):
    pass


class IgnoreNormalizer(PropertyNormalizer):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x
