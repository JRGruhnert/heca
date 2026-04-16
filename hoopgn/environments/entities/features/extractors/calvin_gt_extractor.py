import torch
from dataclasses import dataclass

from hoopgn.environments.properties.features.extractors.extractor import (
    PropertyExtractor,
    PropertyExtractorConfig,
)


@dataclass(kw_only=True)
class CalvinGTExtractorConfig(PropertyExtractorConfig):
    label: str


class CalvinGTExtractor(PropertyExtractor):
    def __init__(self, config: CalvinGTExtractorConfig):
        self.config = config

    def __call__(self, observation) -> torch.Tensor:
        return torch.tensor(observation[self.config.label])
