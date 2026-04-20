import torch
from dataclasses import dataclass

from hoopgn.properties.features.extractors.extractor import (
    PropertyExtractor,
)


class CalvinGTExtractor(PropertyExtractor):
    @dataclass(kw_only=True)
    class Config(PropertyExtractor.Config):
        label: str

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def __call__(self, observation) -> torch.Tensor:
        return torch.tensor(observation[self.cfg.label])
