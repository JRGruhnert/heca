from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
from hoopgn.environments.properties.features.feature import (
    PropertyFeature,
    PropertyFeatureConfig,
)


@dataclass(kw_only=True)
class PropertyEvaluatorConfig(PropertyFeatureConfig):
    pass


class PropertyEvaluator(PropertyFeature):
    def __init__(self, config: PropertyEvaluatorConfig):
        super().__init__(config)
        self.config = config

    @abstractmethod
    def __call__(self, current: torch.Tensor, goal: torch.Tensor) -> bool:
        raise NotImplementedError("Subclasses must implement the __call__ method.")
