from abc import abstractmethod
from dataclasses import dataclass
import torch

from hoopgn.properties.features.feature import (
    PropertyFeature,
    PropertyFeatureConfig,
)


@dataclass(kw_only=True)
class PropertyNormalizerConfig(PropertyFeatureConfig):
    pass


class PropertyNormalizer(PropertyFeature):
    def __init__(
        self,
        config: PropertyNormalizerConfig,
    ):
        super().__init__(config)
        self.config = config

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the processed value of the state as a tensor."""
        raise NotImplementedError("Subclasses must implement the __call__ method.")
