from dataclasses import dataclass

import torch

from hoopgn.properties.features.parameters.parameter import (
    PropertyParameter,
    PropertyParameterConfig,
)
from hoopgn.properties.states.binary_state import (
    BinaryState,
    BinaryStateConfig,
)


@dataclass(kw_only=True)
class BinaryParameterConfig(PropertyParameterConfig):
    binary: BinaryStateConfig = BinaryStateConfig()


class BinaryParameter(PropertyParameter):
    def __init__(self, config: BinaryParameterConfig):
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
        assert isinstance(start, torch.Tensor), "start must be a torch.Tensor"
        assert isinstance(end, torch.Tensor), "end must be a torch.Tensor"
        if self.binary.is_binary(start) and self.binary.is_binary(end):
            if self.binary.is_always_same(start, end):
                candidate = end if reversed else start
                return candidate.mean(dim=0)
        return None  # Not constant enough
