from dataclasses import dataclass, field

import torch

from hoopgn.properties.states.state import State


class BinaryState(State):
    # TODO: This class is weird
    @dataclass(kw_only=True)
    class Config(State.Config):
        values: set[str] = field(default_factory=lambda: {"False", "True"})

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg

    def label(self, x: torch.Tensor) -> str:
        if self.is_binary(x):
            if self.is_lower(x):
                return "False"
            else:
                return "True"
        # NOTE: We dont allow non-binary values. The check is overprotective anyway and it should not happen.
        raise ValueError(
            f"Tensor contains non-binary values: {x}."
            f"This should not happen if the state is used correctly."
        )

    def is_binary(self, x: torch.Tensor) -> bool:
        """Check if the tensor contains only binary values (0 and 1)."""
        return bool(((x == 0.0) | (x == 1.0)).all())

    def is_lower(self, x: torch.Tensor) -> bool:
        """Check if the tensor contains only 0 values."""
        return bool((x == 0.0).all())

    def is_never_equal(self, x: torch.Tensor, y: torch.Tensor) -> bool:
        """Check if two tensors are never equal."""
        return bool((y == (1 - x)).all().item())

    def is_always_same(self, x: torch.Tensor, y: torch.Tensor) -> bool:
        return bool(((y < x) | (x < y)).all().item())
