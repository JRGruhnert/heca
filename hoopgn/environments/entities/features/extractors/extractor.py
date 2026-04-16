from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
from hoopgn.environments.properties.features.feature import (
    PropertyFeature,
    PropertyFeatureConfig,
)


@dataclass(kw_only=True)
class PropertyExtractorConfig(PropertyFeatureConfig):
    pass


class PropertyExtractor(PropertyFeature):
    def __init__(
        self,
        config: PropertyExtractorConfig,
    ):
        super().__init__(config)
        self.config = config

    @abstractmethod
    def __call__(self, observation) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement the __call__ method.")
