from abc import abstractmethod
from dataclasses import dataclass
import torch

from hoopgn.environments.properties.features.feature import PropertyFeature


class PropertyNormalizer(PropertyFeature):
    @dataclass(kw_only=True)
    class Config(PropertyFeature.Config):
        pass

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the processed value of the state as a tensor."""
        raise NotImplementedError("Subclasses must implement the __call__ method.")
