from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(kw_only=True)
class EntityFeatureConfig:
    pass


class EntityFeature(ABC):
    def __init__(
        self,
        config: EntityFeatureConfig,
    ):
        self.config = config

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement the __call__ method.")
