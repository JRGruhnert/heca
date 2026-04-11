from dataclasses import dataclass

import torch

from hoopgn.objects.properties.handlers.normalizers.normalizer import (
    Normalizer,
    NormalizerConfig,
)


@dataclass(kw_only=True)
class IgnoreNormalizerConfig(NormalizerConfig):
    pass


class IgnoreNormalizer(Normalizer):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x
