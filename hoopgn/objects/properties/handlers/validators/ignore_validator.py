from dataclasses import dataclass
import torch

from hoopgn.objects.properties.handlers.validators.validator import (
    StateValidator,
    StateValidatorConfig,
)


@dataclass(kw_only=True)
class IgnoreValidatorConfig(StateValidatorConfig):
    default: float = 0.0


class IgnoreValidator(StateValidator):
    def __init__(self, config: IgnoreValidatorConfig):
        super().__init__(config)
        self.config = config

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> bool:
        return True
