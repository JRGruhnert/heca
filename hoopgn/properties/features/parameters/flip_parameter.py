from dataclasses import dataclass

import torch

from hoopgn.properties.features.parameters.parameter import (
    PropertyParameter,
)
from hoopgn.properties.states.binary_state import (
    BinaryState,
)


class FlipParameter(PropertyParameter):
    @dataclass(kw_only=True)
    class Config(PropertyParameter.Config):
        binary: BinaryState.Config = BinaryState.Config()

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.binary = BinaryState(cfg.binary)

    def hoopgnv1(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        selected_by_tapas: bool = False,
    ) -> torch.Tensor | None:
        """Returns the mean of the given tensor values."""
        assert isinstance(start, torch.Tensor), "start must be a torch.Tensor"
        assert isinstance(end, torch.Tensor), "end must be a torch.Tensor"
        if self.binary.is_binary(start) and self.binary.is_binary(end):
            if self.binary.is_never_equal(start, end):
                return torch.tensor([1.0])  # NOTE Flip state
        return None
