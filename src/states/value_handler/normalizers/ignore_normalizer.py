from dataclasses import dataclass

import torch

from src.states.logic.value_handler.normalizers.normalizer import (
    Normalizer,
    NormalizerConfig,
)


@dataclass
class IgnoreValueConfig(NormalizerConfig):
    pass


class IgnoreValue(Normalizer):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x
