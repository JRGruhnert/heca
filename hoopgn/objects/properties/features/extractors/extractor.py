from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
from hoopgn.objects.properties.features.feature import Feature, FeatureConfig


@dataclass(kw_only=True)
class ExtractorConfig(FeatureConfig):
    pass


class Extractor(Feature):
    def __init__(
        self,
        config: ExtractorConfig,
    ):
        super().__init__(config)
        self.config = config

    @abstractmethod
    def __call__(self, *args, **kwargs) -> torch.Tensor:
        assert all(
            isinstance(arg, torch.Tensor) for arg in args
        ), "All inputs must be torch.Tensor"
        raise NotImplementedError("Subclasses must implement the __call__ method.")
