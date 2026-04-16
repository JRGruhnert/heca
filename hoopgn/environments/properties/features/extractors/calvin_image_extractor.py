import numpy as np
import torch
from dataclasses import dataclass

from hoopgn.environments.properties.features.extractors.extractor import (
    PropertyExtractor,
    PropertyExtractorConfig,
)


@dataclass(kw_only=True)
class CalvinImageExtractorConfig(PropertyExtractorConfig):
    pass


class CalvinImageExtractor(PropertyExtractor):
    def __init__(self, config: CalvinImageExtractorConfig):
        self.config = config

    def __call__(self, observation) -> torch.Tensor:
        # TODO: add image preprocessing here
        raise NotImplementedError("CalvinImageExtractor is not implemented yet.")
