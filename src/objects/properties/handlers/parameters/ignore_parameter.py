from dataclasses import dataclass

import torch

from src.objects.properties.handlers.parameters.parameter import (
    StateParameter,
    StateParameterConfig,
)
from src.objects.properties.binary import Binary, BinaryConfig


@dataclass(kw_only=True)
class IgnoreParameterConfig(StateParameterConfig):
    binary: BinaryConfig = BinaryConfig()


class IgnoreParameter(StateParameter):
    def __init__(self, config: IgnoreParameterConfig):
        super().__init__(config)
        self.config = config
        self.binary = Binary(config.binary)

    def hoopgnv1(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        reversed: bool,
        selected_by_tapas: bool = False,
    ) -> torch.Tensor | None:
        """Returns the mean of the given tensor values."""
        assert isinstance(start, torch.Tensor), "start must be a torch.Tensor"
        assert isinstance(end, torch.Tensor), "end must be a torch.Tensor"
        if self.binary.is_binary(start) and self.binary.is_binary(end):
            if (end == (1 - start)).all():
                return torch.tensor([1.0])  # NOTE Flip state
        return None
