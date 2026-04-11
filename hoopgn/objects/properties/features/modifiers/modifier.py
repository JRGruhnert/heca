from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
from hoopgn.objects.properties.features.feature import Feature, FeatureConfig


@dataclass(kw_only=True)
class ModifierConfig(FeatureConfig):
    pass


class Modifier(Feature):
    def __init__(self, config: ModifierConfig):
        super().__init__(config)
        self.config = config

    @abstractmethod
    def __call__(self, *args, **kwargs) -> torch.Tensor:
        assert all(
            isinstance(arg, torch.Tensor) for arg in args
        ), "All inputs must be torch.Tensor"
        raise NotImplementedError("Subclasses must implement the __call__ method.")
