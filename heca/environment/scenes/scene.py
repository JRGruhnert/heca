from abc import abstractmethod
from dataclasses import dataclass

import numpy as np
from heca.classes.register import Registerable
from heca.entities.entity import Entity

from heca.environment.extractors.calvin_image import ImageExtractor
from heca.misc.td import TDScene

from heca.environment.extractors.extractor import Extractor


class Scene(Registerable):
    @dataclass(kw_only=True)
    class Config(Registerable.Config):
        img_extractor: ImageExtractor.Config | None = None
        gt_extractor: Extractor.Config | None = None

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.converters = {
            k: Converter.create(v) for k, v in self.cfg.converters.items()
        }
        self.extractor = Extractor.create(self.cfg.encoder)

    def to_observation(self, obs) -> tuple[TDScene, bool]:
        formats = {k: self.converters[k](obs) for k in self.converters}
        valid = all(v[1] for v in formats.values())
        return TDScene(formats={k: v[0] for k, v in formats.items()}), valid

    def sample_image(self) -> dict[str, np.ndarray]:
        if "image" not in self.converters:
            raise ValueError("No image converter specified for this scene.")
        return self.converters["image"](obs)

    @abstractmethod
    def step(self, action: np.ndarray) -> TDScene:
        raise NotImplementedError()

    @abstractmethod
    def close(self):
        raise NotImplementedError()

    @abstractmethod
    def sample(self) -> TDScene:
        raise NotImplementedError()

    def entities(self) -> list[Entity.Query]:
        raise NotImplementedError()
