import numpy as np
import torch
from dataclasses import dataclass

from hoopgn.objects.properties.features.extractors.extractor import (
    Extractor,
    ExtractorConfig,
)


@dataclass(kw_only=True)
class GTExtractorConfig(ExtractorConfig):
    pass


class GTExtractor(Extractor):
    def __init__(self, config: GTExtractorConfig):
        self.config = config

    def __call__(self, x: np.ndarray) -> torch.Tensor:
        return torch.tensor(x)
