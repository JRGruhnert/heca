from dataclasses import dataclass

import torch

from src.states.logic.values.value import Value, ValueConfig


@dataclass
class IdentityValueConfig(ValueConfig):
    type_str: str = "IdentityValue"


class IdentityValue(Value):
    """Value converter for discrete states."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Return input value as it is."""
        return x

    def make_input(self, x: torch.Tensor) -> torch.Tensor:
        """Return input value as it is."""
        return self.__call__(x)
