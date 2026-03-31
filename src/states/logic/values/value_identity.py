from dataclasses import dataclass

import torch

from src.states.logic.values.value import Value, ValueHandlerConfig


@dataclass
class IdentityValueConfig(ValueHandlerConfig):
    type_str: str = "IdentityValue"


class IdentityValue(Value):
    """Value converter for discrete states."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Return input value as it is."""
        return x

    def make_input(self, x: torch.Tensor) -> torch.Tensor:
        """Return input value as it is."""
        return self.__call__(x)
