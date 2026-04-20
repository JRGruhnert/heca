from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch

from hoopgn.properties.features.feature import (
    PropertyFeature,
    PropertyFeatureConfig,
)


@dataclass(kw_only=True)
class PropertyValidatorConfig(PropertyFeatureConfig):
    pass


class PropertyValidator(PropertyFeature):
    def __init__(
        self,
        config: PropertyValidatorConfig,
    ):
        super().__init__(config)
        self.config = config

    @abstractmethod
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> bool:
        raise NotImplementedError("Subclasses must implement the __call__ method.")
