from abc import abstractmethod
from dataclasses import dataclass
import torch

from hoopgn.objects.properties.features.feature import Feature, FeatureConfig


@dataclass(kw_only=True)
class NormalizerConfig(FeatureConfig):
    pass


class Normalizer(Feature):
    def __init__(
        self,
        config: NormalizerConfig,
    ):
        super().__init__(config)
        self.config = config

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the processed value of the state as a tensor."""
        raise NotImplementedError("Subclasses must implement the __call__ method.")
