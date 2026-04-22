import torch


class Binary:

    @staticmethod
    def is_binary(x: torch.Tensor) -> bool:
        """Check if the tensor contains only binary values (0 and 1)."""
        return bool(((x == 0.0) | (x == 1.0)).all())

    @staticmethod
    def is_lower(x: torch.Tensor) -> bool:
        """Check if the tensor contains only 0 values."""
        return bool((x == 0.0).all())

    @staticmethod
    def is_never_equal(x: torch.Tensor, y: torch.Tensor) -> bool:
        """Check if two tensors are never equal."""
        return bool((y == (1 - x)).all().item())

    @staticmethod
    def is_always_same(x: torch.Tensor, y: torch.Tensor) -> bool:
        """Check if two tensors are always the same."""
        return bool((x == y).all().item())
