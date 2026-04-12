from dataclasses import dataclass

import torch

from hoopgn.entities.properties.features.parameters.parameter import (
    PropertyParameter,
    PropertyParameterConfig,
)
from hoopgn.entities.properties.states.binary_state import (
    BinaryState,
    BinaryStateConfig,
)


@dataclass(kw_only=True)
class IgnoreParameterConfig(PropertyParameterConfig):
    binary: BinaryStateConfig = BinaryStateConfig()


class IgnoreParameter(PropertyParameter):
    def __init__(self, config: IgnoreParameterConfig):
        super().__init__(config)
        self.config = config
        self.binary = BinaryState(config.binary)

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
