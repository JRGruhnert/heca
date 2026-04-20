from abc import abstractmethod
from dataclasses import dataclass
import torch
from hoopgn.properties.features.feature import PropertyFeature


class PropertyModifier(PropertyFeature):
    @dataclass(kw_only=True)
    class Config(PropertyFeature.Config):
        pass

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg

    @abstractmethod
    def __call__(self, *args, **kwargs) -> torch.Tensor:
        assert all(
            isinstance(arg, torch.Tensor) for arg in args
        ), "All inputs must be torch.Tensor"
        raise NotImplementedError("Subclasses must implement the __call__ method.")
