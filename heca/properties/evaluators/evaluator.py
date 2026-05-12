from abc import abstractmethod
from dataclasses import dataclass
import torch

from heca.classes.config import Configurable


class PropertyEvaluator(Configurable):

    @abstractmethod
    def __call__(self, x: torch.Tensor, y: torch.Tensor, distance: float) -> bool:
        raise NotImplementedError()
