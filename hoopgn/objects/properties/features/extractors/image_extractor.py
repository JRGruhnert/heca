import numpy as np
import torch
from dataclasses import dataclass

from hoopgn.objects.properties.features.extractors.extractor import (
    Extractor,
    ExtractorConfig,
)


@dataclass(kw_only=True)
class ImageExtractorConfig(ExtractorConfig):
    pass


class ImageExtractor(Extractor):
    def __init__(self, config: ImageExtractorConfig):
        self.config = config

    def __call__(self, x: np.ndarray) -> torch.Tensor:
        # TODO: add image preprocessing here
        return torch.tensor(x)
