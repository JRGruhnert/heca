from abc import abstractmethod
from dataclasses import dataclass
import torch
from hoopgn.misc.classes import ConfigClass


class PropertyRuler(ConfigClass):
    @dataclass(kw_only=True)
    class Config(ConfigClass.Config):
        pass

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> float:
        value = self.distance(a, b)
        assert isinstance(value, float), "Distance must be a float"
        assert 0.0 <= value <= 1.0, "Distance must be in [0.0, 1.0]"
        return value

    @abstractmethod
    def distance(self, a: torch.Tensor, b: torch.Tensor) -> float:
        raise NotImplementedError("Subclasses must implement the distance method.")
