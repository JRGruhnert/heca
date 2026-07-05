from abc import abstractmethod
import torch

from heca.misc.base import Configurable


class PropertyEvaluator(Configurable):

    @abstractmethod
    def __call__(self, x: torch.Tensor, y: torch.Tensor, distance: float) -> bool:
        raise NotImplementedError()
