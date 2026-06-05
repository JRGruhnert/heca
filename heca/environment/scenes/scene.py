import re
import abc
import torch
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from PIL import Image


from heca.entities.entity import Entity
from heca.misc.td import TDScene, TDSceneImages
from heca.misc.base import Persistable


class Scene(Persistable):
    @dataclass(kw_only=True)
    class Config(Persistable.Config):
        label: str
        folder: str = "samples"

    def __init__(self, cfg: Config):
        self.cfg = cfg

        self.kp_references: dict[Entity, tuple[Image.Image, int, int, int, int]] = {}
        self.state_references: dict[Entity, dict[str, list[Image.Image]]] = {}

    def from_internal(self, data) -> tuple[TDScene, TDSceneImages]:
        tdscene = self.heca_td(data)
        tdimage = self.to_td_scene_images(data)
        return tdscene, tdimage

    def step(self, action: np.ndarray) -> tuple[TDScene, TDSceneImages]:
        obs = self._step(action)
        return self.from_internal(obs)

    @abc.abstractmethod
    def _step(self, action: np.ndarray) -> Any:
        raise NotImplementedError()

    @abc.abstractmethod
    def sample_task(
        self,
    ) -> tuple[
        tuple[TDScene, TDSceneImages],
        tuple[TDScene, TDSceneImages],
    ]:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_cursor(self, obs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    @abc.abstractmethod
    def heca_td(self, obs) -> TDScene:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def entities(self) -> list[Entity]:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def cursor(self) -> Entity:
        raise NotImplementedError()

    @abc.abstractmethod
    def to_td_scene_images(self, obs) -> TDSceneImages:
        raise NotImplementedError()

    @abc.abstractmethod
    def sample_image(self) -> np.ndarray:
        raise NotImplementedError()

    def _load(self, path: Path):
        dc_pattern = re.compile(rf"xk(\d+)_yk(\d+)_xs(\d+)_ys(\d+)\.png")
        sample_postfix = r"_sample(\d+)\.png"
        for entity in self.entities:
            edir = path / entity.cfg.label
            for state in entity.cfg.states:
                self.state_references[entity][state] = []
                state_pattern = re.compile(rf"{re.escape(state)}{sample_postfix}")
                for file in edir.glob(f"{state}_sample*.png"):
                    if state_pattern.fullmatch(file.name):
                        self.state_references[entity][state].append(
                            Image.open(file),
                        )
            files = list(edir.glob(f"xk*_yk*_xs*_ys*.png"))
            assert files is not None
            assert len(files) == 1
            file = files[0]
            match = dc_pattern.fullmatch(file.name)
            if match:
                self.kp_references[entity] = (
                    Image.open(file),
                    int(match.group(1)),
                    int(match.group(2)),
                    int(match.group(3)),
                    int(match.group(4)),
                )

    def _save(self, path: Path):
        for entity in self.entities:
            entity_dir = path / entity.cfg.label
            entity_dir.mkdir(parents=True, exist_ok=True)
            for state, samples in self.state_references[entity].items():
                for idx, img in enumerate(samples):
                    img.save(
                        entity_dir / f"{state}_sample{idx}.png"
                    )  # e.g., "open_sample0.png"
            img, x1, y1, x2, y2 = self.kp_references[entity]
            img.save(entity_dir / f"xk{x1}_yk{y1}_xs{x2}_ys{y2}.png")

    @abc.abstractmethod
    def test(self):
        raise NotImplementedError()
