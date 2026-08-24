import re
import abc
import h5py
import torch
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from PIL import Image


from heca.data.data import DCScene, TDImage
from heca.data.entity import Entity
from heca.misc.base import Persistable


@dataclass(kw_only=True, slots=True)
class SceneFeedback:
    terminal: bool
    reward: float
    truncated: bool


class Scene(Persistable):
    @dataclass(kw_only=True)
    class Config(Persistable.Config):
        folder: str = "scenes"
        width: int = 256
        height: int = 256
        visualize: bool = False

    def __init__(self, cfg: Config):
        self.cfg = cfg

        self.kp_references: dict[str, tuple[Image.Image, int, int, int, int]] = {}
        self.state_references: dict[str, dict[int, list[Image.Image]]] = {}

    def from_internal(self, data) -> tuple[DCScene, TDImage, np.ndarray]:
        tdscene = self.to_dc_scene(data)
        tdimage = self.to_td_image(data)
        npimage = self.to_np_image(data)
        return tdscene, tdimage, npimage

    def virtual_evaluation(
        self, x: DCScene, y: DCScene
    ) -> tuple[DCScene, SceneFeedback]:
        raise NotImplementedError

    def step(self, action: np.ndarray) -> tuple[DCScene, TDImage, SceneFeedback]:
        obs, fb = self._step(action)
        tdscene, tdimage, _ = self.from_internal(obs)
        return tdscene, tdimage, fb

    def step_virt(
        self, x: DCScene, y: DCScene, elabels: list[str]
    ) -> tuple[DCScene, TDImage, SceneFeedback]:
        obs, fb = self._step_virt(x, y, elabels)
        tdscene, tdimage, _ = self.from_internal(obs)
        return tdscene, tdimage, fb

    def step_vis(self, action: np.ndarray) -> tuple[DCScene, TDImage, np.ndarray]:
        obs, _ = self._step(action)
        return self.from_internal(obs)

    @abc.abstractmethod
    def _step(self, action: np.ndarray) -> tuple[Any, SceneFeedback]:
        raise NotImplementedError()

    @abc.abstractmethod
    def _step_virt(
        self, x: DCScene, y: DCScene, elabels: list[str]
    ) -> tuple[Any, SceneFeedback]:
        raise NotImplementedError()

    @abc.abstractmethod
    def sample_task(self) -> tuple[
        tuple[DCScene, TDImage],
        tuple[DCScene, TDImage],
    ]:
        raise NotImplementedError()

    def sample_task_vis(self) -> tuple[
        tuple[DCScene, TDImage, np.ndarray],
        tuple[DCScene, TDImage, np.ndarray],
    ]:
        raise NotImplementedError()

    def get_ee(self, obs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    @abc.abstractmethod
    def to_dc_scene(self, obs) -> DCScene:
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
        with_images: bool = True,
    ) -> tuple[list[list[DCScene]], list[list[TDImage]]]:
        raise NotImplementedError()

    def _load(self, path: Path) -> bool:
        dc_pattern = re.compile(rf"xk(\d+)_yk(\d+)_xs(\d+)_ys(\d+)\.png")
        sample_postfix = r"_sample(\d+)\.png"
        for label, entity in self.entities.items():
            edir = path / label
            self.state_references[label] = {}
            for idx in range(entity.cfg.n_states):
                self.state_references[label][idx] = []
                state_pattern = re.compile(rf"{idx}{sample_postfix}")
                for file in edir.glob(f"{idx}_sample*.png"):
                    if state_pattern.fullmatch(file.name):
                        self.state_references[label][idx].append(
                            Image.open(file),
                        )
            files = list(edir.glob("xk*_yk*_xs*_ys*.png"))
            if files:
                assert len(files) == 1
                file = files[0]
                match = dc_pattern.fullmatch(file.name)
                if match:
                    self.kp_references[label] = (
                        Image.open(file),
                        int(match.group(1)),
                        int(match.group(2)),
                        int(match.group(3)),
                        int(match.group(4)),
                    )
        return True

    def _save(self, path: Path) -> bool:
        for label in self.entities.keys():
            entity_dir = path / label
            entity_dir.mkdir(parents=True, exist_ok=True)
            for state, samples in self.state_references[label].items():
                for idx, img in enumerate(samples):
                    img.save(entity_dir / f"{state}_sample{idx}.png")
            img, x1, y1, x2, y2 = self.kp_references[label]
            img.save(entity_dir / f"xk{x1}_yk{y1}_xs{x2}_ys{y2}.png")
        return True

    @property
    def description(self) -> str:
        raise NotImplementedError()

    @property
    def entities(self) -> dict[str, Entity]:
        raise NotImplementedError()

    @property
    def dataset_path(self) -> str:
        raise NotImplementedError()

    def demo_auto_extract(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def normalize_position(self, pos) -> np.ndarray:
        raise NotImplementedError

    def unnormalize_position(self, pos) -> np.ndarray:
        raise NotImplementedError
