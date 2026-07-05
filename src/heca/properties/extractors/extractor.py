from abc import abstractmethod
from dataclasses import dataclass
import torch

from heca.misc.base import Configurable


class PropertyExtractor(Configurable):
    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        pass

    def __init__(self, cfg: Config):
        self.cfg = cfg

    @abstractmethod
    def __call__(self, observation) -> torch.Tensor:
        raise NotImplementedError()
