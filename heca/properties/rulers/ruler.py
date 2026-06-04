from abc import abstractmethod
import abc
import torch
from heca.misc.base import Configurable


class PropertyRuler(Configurable, abc.ABC):

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> float:
        value = self.distance(a, b)
        assert isinstance(value, float), "Distance must be a float"
        assert 0.0 <= value <= 1.0, "Distance must be in [0.0, 1.0]"
        return value

    @abc.abstractmethod
    def distance(self, a: torch.Tensor, b: torch.Tensor) -> float:
        raise NotImplementedError("Subclasses must implement the distance method.")
