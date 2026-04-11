from dataclasses import dataclass
import torch

from hoopgn.objects.properties.features.validators.validator import (
    PropertyValidator,
    PropertyValidatorConfig,
)


@dataclass(kw_only=True)
class SkipValidatorConfig(PropertyValidatorConfig):
    default: float = 0.0


class SkipValidator(PropertyValidator):
    def __init__(self, config: SkipValidatorConfig):
        super().__init__(config)
        self.config = config

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> bool:
        return True
