import torch
from dataclasses import dataclass

from hoopgn.entities.properties.extractors.extractor import (
    PropertyExtractor,
)


class CImageExtractor(PropertyExtractor):
    @dataclass(kw_only=True)
    class Config(PropertyExtractor.Config):
        pass

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg

    def __call__(self, observation) -> torch.Tensor:
        # TODO: add image preprocessing here
        raise NotImplementedError()
