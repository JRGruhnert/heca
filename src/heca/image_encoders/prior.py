from dataclasses import dataclass

import numpy as np
import torch


@dataclass(kw_only=True)
class PriorConfig:
    """How strongly the previous frames constrain the current one."""

    #: spread of the belief on the descriptor grid, in grid cells
    sigma: float = 3.0
    #: similarity a detection needs to be trusted on its own
    threshold: float = 0.2


class KeypointPrior:
    def __init__(self, config: PriorConfig):
        self.cfg = config
        self.pixel: tuple[float, float] | None = None  # (row, column) on the grid

    def reset(self) -> None:
        self.pixel = None

    @property
    def has_belief(self) -> bool:
        return self.pixel is not None

    def grid_position(self, height: int, width: int) -> tuple[float, float]:
        """The belief in grid cells, from the normalised coordinates it is kept in."""
        assert self.pixel is not None
        return (
            (self.pixel[0] + 1.0) * 0.5 * (height - 1),
            (self.pixel[1] + 1.0) * 0.5 * (width - 1),
        )

    def predict(self, height: int, width: int) -> torch.Tensor | None:
        """The prior over the descriptor grid, or ``None`` without a belief."""
        if self.pixel is None:
            return None
        centre = self.grid_position(height, width)
        rows = torch.arange(height, dtype=torch.float32).reshape(-1, 1)
        cols = torch.arange(width, dtype=torch.float32).reshape(1, -1)
        distance2 = (rows - centre[0]) ** 2 + (cols - centre[1]) ** 2
        return torch.exp(-0.5 * distance2 / self.cfg.sigma**2)

    def update(self, pixel: tuple[float, float]) -> None:
        self.pixel = (float(pixel[0]), float(pixel[1]))

    def hold(self) -> tuple[float, float] | None:
        """Where the keypoint was, for a frame that carries no information."""
        return self.pixel


def fill_gaps(
    values: list[np.ndarray | None],
) -> list[np.ndarray]:
    known = [i for i, value in enumerate(values) if value is not None]
    if not known:
        raise ValueError("an entity with no detected frame at all cannot be filled")
    filled: list[np.ndarray] = []
    for index, value in enumerate(values):
        if value is not None:
            filled.append(np.asarray(value, dtype=np.float64))
            continue
        before = [i for i in known if i < index]
        after = [i for i in known if i > index]
        if not before:
            filled.append(np.asarray(values[after[0]], dtype=np.float64))
        elif not after:
            filled.append(np.asarray(values[before[-1]], dtype=np.float64))
        else:
            lo, hi = before[-1], after[0]
            weight = (index - lo) / (hi - lo)
            filled.append(
                (1.0 - weight) * np.asarray(values[lo], dtype=np.float64)
                + weight * np.asarray(values[hi], dtype=np.float64)
            )
    return filled
