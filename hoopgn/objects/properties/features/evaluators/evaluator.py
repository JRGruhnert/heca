from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
from hoopgn.objects.properties.features.feature import Feature, FeatureConfig


@dataclass(kw_only=True)
class StateEvaluatorConfig(FeatureConfig):
    pass


class StateEvaluator(Feature):
    def __init__(self, config: StateEvaluatorConfig):
        super().__init__(config)
        self.config = config

    @abstractmethod
    def __call__(self, current: torch.Tensor, goal: torch.Tensor) -> bool:
        raise NotImplementedError("Subclasses must implement the __call__ method.")
