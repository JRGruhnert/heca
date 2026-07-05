import re
import abc
import h5py
import torch
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from PIL import Image


from heca.misc.entity import Entity
from heca.misc.td import TDImage, TDScene
from heca.misc.base import Persistable


class Scene(Persistable):
    @dataclass(kw_only=True)
    class Config(Persistable.Config):
        folder: str = "scenes"
        width: int = 256
        height: int = 256

    def __init__(self, cfg: Config):
        self.cfg = cfg

        self.kp_references: dict[str, tuple[Image.Image, int, int, int, int]] = {}
        self.state_references: dict[str, dict[str, list[Image.Image]]] = {}

    def from_internal(self, data) -> tuple[TDScene, TDImage, np.ndarray]:
        tdscene = self.to_td_scene(data)
        tdimage = self.to_td_image(data)
        npimage = self.to_np_image(data)
        return tdscene, tdimage, npimage

    def step(self, action: np.ndarray) -> tuple[TDScene, TDImage, float, bool, bool]:
        obs, reward, done, truncated = self._step(action)
        tdscene, tdimage, _ = self.from_internal(obs)
        return tdscene, tdimage, reward, done, truncated

    def step_vis(self, action: np.ndarray) -> tuple[TDScene, TDImage, np.ndarray]:
        obs, _, _, _ = self._step(action)
        return self.from_internal(obs)

    @abc.abstractmethod
    def _step(self, action: np.ndarray) -> tuple[Any, float, bool, bool]:
        raise NotImplementedError()

    @abc.abstractmethod
    def sample_task(self) -> tuple[
        tuple[TDScene, TDImage],
        tuple[TDScene, TDImage],
    ]:
        raise NotImplementedError()

    def sample_task_vis(self) -> tuple[
        tuple[TDScene, TDImage, np.ndarray],
        tuple[TDScene, TDImage, np.ndarray],
    ]:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_ee(self, obs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    @abc.abstractmethod
    def to_td_scene(self, obs) -> TDScene:
        raise NotImplementedError()

    @abc.abstractmethod
    def to_td_image(self, obs) -> TDImage:
        raise NotImplementedError()

    @abc.abstractmethod
    def to_np_image(self, obs) -> np.ndarray:
        raise NotImplementedError()

    @abc.abstractmethod
    def load_dataset(
        self,
        file: h5py.File,
        selections: list[int] | None = None,
        only_conditions: bool = False,
    ) -> tuple[list[list[TDScene]], list[list[TDImage]]]:
        raise NotImplementedError()

    def _load(self, path: Path, tag: str):
        dc_pattern = re.compile(rf"xk(\d+)_yk(\d+)_xs(\d+)_ys(\d+)\.png")
        sample_postfix = r"_sample(\d+)\.png"
        for entity in self.entities:
            edir = path / tag / entity.cfg.label
            self.state_references[entity.cfg.label] = {}
            for state in entity.cfg.states:
                self.state_references[entity.cfg.label][state] = []
                state_pattern = re.compile(rf"{re.escape(state)}{sample_postfix}")
                for file in edir.glob(f"{state}_sample*.png"):
                    if state_pattern.fullmatch(file.name):
                        self.state_references[entity.cfg.label][state].append(
                            Image.open(file),
                        )
            files = list(edir.glob(f"xk*_yk*_xs*_ys*.png"))
            assert files is not None
            assert len(files) == 1
            file = files[0]
            match = dc_pattern.fullmatch(file.name)
            if match:
                self.kp_references[entity.cfg.label] = (
                    Image.open(file),
                    int(match.group(1)),
                    int(match.group(2)),
                    int(match.group(3)),
                    int(match.group(4)),
                )

    def _save(self, path: Path, tag: str):
        for entity in self.entities:
            entity_dir = path / tag / entity.cfg.label
            entity_dir.mkdir(parents=True, exist_ok=True)
            for state, samples in self.state_references[entity.cfg.label].items():
                for idx, img in enumerate(samples):
                    img.save(
                        entity_dir / f"{state}_sample{idx}.png"
                    )  # e.g., "open_sample0.png"
            img, x1, y1, x2, y2 = self.kp_references[entity.cfg.label]
            img.save(entity_dir / f"xk{x1}_yk{y1}_xs{x2}_ys{y2}.png")

    @property
    @abc.abstractmethod
    def description(self) -> str:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def entities(self) -> list[Entity]:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def ee(self) -> Entity:
        raise NotImplementedError()
