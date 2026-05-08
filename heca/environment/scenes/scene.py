from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from heca.classes.register import Registerable
from heca.entities.entity import Entity

from heca.environment.extractors.image_extractor import ImageExtractor
from heca.misc.td import TDEntities, TDProperties, TDScene
from tensordict import TensorDict


class Scene(Registerable):
    @dataclass(kw_only=True)
    class Config(Registerable.Config):
        extractor: ImageExtractor.Config = ImageExtractor.Config()
        gt: bool = False
        v1_compatibility: bool = False

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.extractor = ImageExtractor.create(self.cfg.extractor)

    def sample_images(self) -> dict[str, torch.Tensor]:
        obs = self._reset()
        return self.image_tensors(obs)

    def reset(self) -> TDScene:
        obs = self._reset()
        if self.cfg.gt:
            d = {
                "tapas": self.tapas_td(obs),
                "heca": self.heca_td(obs),
            }
        else:
            rgb = self.image_tensors(obs)
            extracted = self.extractor(rgb)
            d = {
                "tapas": self.tapas_td(obs, extracted),
                "heca": extracted,
            }
        if self.cfg.v1_compatibility:
            d["v1"] = self.v1_td(obs)
        return TDScene(d)

    def step(self, action: np.ndarray) -> TDScene:
        obs = self._step(action)
        return self.tapas_td(obs)

    def sample(self) -> TDScene:
        x = self.reset()
        while self.is_bad_sample(x):
            x = self.reset()
        return x

    def is_bad_sample(self, obs: TDScene) -> bool:
        return False  # By default, no sample is bad. Override in specific scenes if needed.

    def entities(self) -> list[Entity.Query]:
        raise NotImplementedError()

    def tapas_td(self, obs, extracted: TDEntities | None = None) -> TensorDict:
        raise NotImplementedError()

    def heca_td(self, obs) -> TDEntities:
        raise NotImplementedError()

    def v1_td(self, obs) -> TDProperties:
        raise NotImplementedError()

    def image_tensors(self, obs) -> dict[str, torch.Tensor]:
        raise NotImplementedError()

    def _reset(self) -> Any:
        raise NotImplementedError()

    def _step(self, action: np.ndarray) -> Any:
        raise NotImplementedError()
