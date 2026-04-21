from dataclasses import dataclass

import torch


class Binary:
    @dataclass(kw_only=True)
    class Validator:
        pass

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
        """Check if two tensors are always the same."""
        return bool((x == y).all().item())
