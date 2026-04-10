from dataclasses import dataclass

import torch

from src.objects.properties.handlers.parameters.parameter import (
    StateParameter,
    StateParameterConfig,
)
from src.objects.properties.binary import Binary, BinaryConfig


@dataclass(kw_only=True)
class BinaryParameterConfig(StateParameterConfig):
    binary: BinaryConfig = BinaryConfig()


class BinaryParameter(StateParameter):
    def __init__(self, config: BinaryParameterConfig):
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
        assert isinstance(start, torch.Tensor), "start must be a torch.Tensor"
        assert isinstance(end, torch.Tensor), "end must be a torch.Tensor"
        if self.binary.is_binary(start) and self.binary.is_binary(end):
            candidate = end if reversed else start
            if ((end < start) | (start < end)).all():
                return candidate.mean(dim=0)
        return None  # Not constant enough
