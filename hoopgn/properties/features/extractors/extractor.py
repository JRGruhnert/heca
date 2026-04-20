from abc import abstractmethod
from dataclasses import dataclass
import torch
from hoopgn.properties.features.feature import PropertyFeature


class PropertyExtractor(PropertyFeature):
    @dataclass(kw_only=True)
    class Config(PropertyFeature.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        self.config = config

    @abstractmethod
    def __call__(self, observation) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement the __call__ method.")
