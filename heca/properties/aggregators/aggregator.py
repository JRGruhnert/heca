from abc import abstractmethod
from dataclasses import dataclass
import torch

from heca.misc.base import Configurable


class PropertyAggregator(Configurable):
    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        pass

    def __init__(self, cfg: Config):
        self.cfg = cfg

    @abstractmethod
    def __call__(
        self, x: torch.Tensor, y: torch.Tensor, values: list[torch.Tensor]
    ) -> torch.Tensor:
        raise NotImplementedError()
