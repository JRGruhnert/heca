from abc import abstractmethod
from dataclasses import dataclass
import torch
from hoopgn.environments.properties.features.feature import PropertyFeature


class PropertyRuler(PropertyFeature):
    @dataclass(kw_only=True)
    class Config(PropertyFeature.Config):
        pass

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> float:
        value = self.distance(a, b)
        assert isinstance(value, float), "Distance must be a float"
        assert 0.0 <= value <= 1.0, "Distance must be in [0.0, 1.0]"
        return value

    @abstractmethod
    def distance(self, a: torch.Tensor, b: torch.Tensor) -> float:
        raise NotImplementedError("Subclasses must implement the distance method.")
