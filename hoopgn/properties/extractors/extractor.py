from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
import torch
from hoopgn.properties.property import Property
from hoopgn.misc.classes import ConfigClass


class PropertyExtractor(ConfigClass):
    @dataclass(kw_only=True)
    class Config(ConfigClass.Config):
        pass

    def __init__(self, cfg: Config):
        self.cfg = cfg

    @abstractmethod
    def __call__(self, observation) -> torch.Tensor:
        raise NotImplementedError()
