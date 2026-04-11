from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(kw_only=True)
class FeatureConfig:
    pass


class Feature(ABC):
    def __init__(
        self,
        config: FeatureConfig,
    ):
        self.config = config

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement the __call__ method.")
