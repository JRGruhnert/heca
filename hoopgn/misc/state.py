from abc import abstractmethod
from dataclasses import dataclass

import torch

from hoopgn.classes import ConfigurableClass
from hoopgn.misc.area import Area


class State(ConfigurableClass):
    @dataclass(kw_only=True)
    class Config(ConfigurableClass.Config):
        values: set[str]

    def __init__(self, cfg: Config):
        self.cfg = cfg
        assert len(cfg.values) > 0, "State must have at least one value."
        assert None not in cfg.values, "State values cannot be None."

        self.one_hots: dict[str, torch.Tensor] = {
            label: self.make_one_hot(label) for label in cfg.values
        }
        self.zeros = self.make_zeros()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        label = self.label(x)
        if label in self.cfg.values:
            return self.one_hots.get(label, self.zeros)
        else:
            return self.zeros

    @abstractmethod
    def label(self, x: torch.Tensor) -> str | None:
        raise NotImplementedError()

    def make_zeros(self) -> torch.Tensor:
        return torch.zeros(len(self.cfg.values), dtype=torch.float32)

    def make_one_hot(self, label: str) -> torch.Tensor:
        assert label is not None, "Label cannot be None."
        assert label in self.cfg.values, "Label must be in state values."
        one_hot = self.make_zeros()
        index = list(self.cfg.values).index(label)
        one_hot[index] = 1.0
        return one_hot

    @classmethod
    def from_area_config(cls, area_cfg: Area.Config) -> "State":
        return cls(
            cls.Config(values=area_cfg.labels),
        )
