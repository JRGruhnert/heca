from dataclasses import dataclass

import torch

from src.states.logic.value_cnd import ValueCondition, ValueConfig


@dataclass
class IdentityValueConfig(ValueConfig):
    type_str: str = "IdentityValue"


class IdentityValue(ValueCondition):
    """Value converter for discrete states."""

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """Return input value as it is."""
        return x

    def make_input(self, x: torch.Tensor) -> torch.Tensor:
        """Return input value as it is."""
        return self.value(x)
