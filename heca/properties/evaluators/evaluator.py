from abc import abstractmethod
from dataclasses import dataclass
import torch

from heca.misc.classes import ConfigClass


class PropertyEvaluator(ConfigClass):
    @dataclass(kw_only=True)
    class Config(ConfigClass.Config):
        pass

    def __init__(self, cfg: Config):
        self.cfg = cfg

    @abstractmethod
    def __call__(self, x: torch.Tensor, y: torch.Tensor, distance: float) -> bool:
        raise NotImplementedError()
