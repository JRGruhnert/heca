from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from heca.classes.register import Registerable
from heca.entities.entity import Entity

from heca.environment.image_extractor import ImageExtractor
from heca.misc.td import TDCamRecordings, TDEntities, TDProperties, TDScene
from tensordict import TensorDict


class Scene(Registerable):
    @dataclass(frozen=True, kw_only=True)
    class Query(Registerable.Query):
        label: str

    @dataclass(kw_only=True)
    class Config(Registerable.Config):
        extractor: ImageExtractor.Query
        gt: bool = False
        v1_compatibility: bool = False
        img_size: tuple[int, int]

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.extractor = ImageExtractor.load(self.cfg.extractor)

    def sample_images(self) -> TDCamRecordings:
        obs = self._reset()
        return self.image_tensors2(obs)

    def reset(self) -> TDScene:
        obs = self._reset()
        return self.make_full_td(obs)

    def step(self, action: np.ndarray) -> Any:
        return self._step(action)

    def make_full_td(self, obs) -> TDScene:
        if self.cfg.gt:
            data = {
                "tapas": self.tapas_td(obs),
                "heca": self.heca_td(obs),
            }
        else:
            recordings = self.image_tensors2(obs)
            extracted = self.extractor(recordings)
            data = {
                "tapas": self.tapas_td(obs, extracted),
                "heca": extracted,
            }
        if self.cfg.v1_compatibility:
            data["v1"] = self.v1_td(obs)
        return TDScene(data)

    def sample(self) -> TDScene:
        x = self.reset()
        while self.is_bad_sample(x):
            x = self.reset()
        return x

    def is_bad_sample(self, obs: TDScene) -> bool:
        return False  # By default, no sample is bad. Override in specific scenes if needed.

    @property
    def entities(self) -> list[Entity]:
        raise NotImplementedError()

    def tapas_td(self, obs, extracted: TDEntities | None = None) -> TensorDict:
        raise NotImplementedError()

    def heca_td(self, obs) -> TDEntities:
        raise NotImplementedError()

    def v1_td(self, obs) -> TDProperties:
        raise NotImplementedError()

    def image_tensors(self, obs) -> dict[str, torch.Tensor]:
        raise NotImplementedError()

    def image_tensors2(self, obs) -> TDCamRecordings:
        raise NotImplementedError()

    def _reset(self) -> Any:
        raise NotImplementedError()

    def _step(self, action: np.ndarray) -> Any:
        raise NotImplementedError()
