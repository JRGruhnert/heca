from dataclasses import dataclass
import torch

from hoopgn.properties.features.validators.validator import (
    PropertyValidator,
)


class SkipValidator(PropertyValidator):
    @dataclass(kw_only=True)
    class Config(PropertyValidator.Config):
        default: float = 0.0

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> bool:
        return True
