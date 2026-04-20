from abc import abstractmethod
from dataclasses import dataclass
import torch

from hoopgn.environments.properties.features.feature import PropertyFeature


class PropertyValidator(PropertyFeature):
    @dataclass(kw_only=True)
    class Config(PropertyFeature.Config):
        pass

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg

    @abstractmethod
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> bool:
        raise NotImplementedError("Subclasses must implement the __call__ method.")
