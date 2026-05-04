from abc import abstractmethod
from dataclasses import dataclass
import torch

from heca.classes.config import Configurable


class PropertyNormalizer(Configurable):
    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        pass

    def __init__(self, cfg: Config):
        self.cfg = cfg

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the processed value of the state as a tensor."""
        raise NotImplementedError("Subclasses must implement the __call__ method.")
