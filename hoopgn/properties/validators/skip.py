from dataclasses import dataclass
import torch

from hoopgn.properties.validators.validator import (
    PropertyValidator,
)


class DefaultValidator(PropertyValidator):
    @dataclass(kw_only=True)
    class Config(PropertyValidator.Config):
        label: str = "default"

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> bool:
        """A default validator that always returns True"""
        return True
