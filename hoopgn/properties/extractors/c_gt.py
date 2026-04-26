import torch
from dataclasses import dataclass

from hoopgn.properties.extractors.extractor import (
    PropertyExtractor,
)


class CGTExtractor(PropertyExtractor):
    @dataclass(kw_only=True)
    class Config(PropertyExtractor.Config):
        field_name: str

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def __call__(self, observation) -> torch.Tensor:
        return torch.tensor(observation[self.cfg.field_name])
