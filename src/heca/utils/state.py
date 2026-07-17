from dataclasses import dataclass
from typing import Callable

import torch

from heca.misc.base import Configurable


class State(Configurable):
    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        values: set[str]
        labeling: Callable[[torch.Tensor], str | None]

    def __init__(self, cfg: Config):
        self.cfg = cfg
        assert len(cfg.values) > 0, "State must have at least one value."
        assert None not in cfg.values, "State values cannot be None."

        self.one_hots: dict[str, torch.Tensor] = {
            label: self.make_one_hot(label) for label in cfg.values
        }
        self.zeros = self.make_zeros()
        self.labeling = cfg.labeling

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        label = self.labeling(x)
        if label in self.cfg.values:
            return self.one_hots.get(label, self.zeros)
        else:
            return self.zeros

    def make_zeros(self) -> torch.Tensor:
        return torch.zeros(len(self.cfg.values), dtype=torch.float32)

    def make_one_hot(self, label: str) -> torch.Tensor:
        assert label is not None, "Label cannot be None."
        assert label in self.cfg.values, "Label must be in state values."
        one_hot = self.make_zeros()
        index = list(self.cfg.values).index(label)
        one_hot[index] = 1.0
        return one_hot

    def one_hot_from_idx(self, idx: int) -> torch.Tensor:
        assert 0 <= idx < len(self.cfg.values), "Index out of bounds."
        one_hot = self.make_zeros()
        one_hot[idx] = 1.0
        return one_hot
