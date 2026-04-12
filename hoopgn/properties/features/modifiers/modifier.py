from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
from hoopgn.properties.features.feature import (
    PropertyFeature,
    PropertyFeatureConfig,
)


@dataclass(kw_only=True)
class PropertyModifierConfig(PropertyFeatureConfig):
    pass


class PropertyModifier(PropertyFeature):
    def __init__(self, config: PropertyModifierConfig):
        super().__init__(config)
        self.config = config

    @abstractmethod
    def __call__(self, *args, **kwargs) -> torch.Tensor:
        assert all(
            isinstance(arg, torch.Tensor) for arg in args
        ), "All inputs must be torch.Tensor"
        raise NotImplementedError("Subclasses must implement the __call__ method.")
