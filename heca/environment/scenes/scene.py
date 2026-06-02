import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from heca.classes.register import Registerable
from heca.entities.entity import Entity
from heca.misc.td import TDScene, TDSceneImages


class Scene(Registerable):

    @dataclass(kw_only=True)
    class Config(Registerable.Config):
        pass

    def from_internal(self, data) -> tuple[TDScene, TDSceneImages]:
        tdscene = self.heca_td(data)
        tdimage = self.to_td_scene_vision(data)
        return tdscene, tdimage

    def reset(self) -> tuple[TDScene, TDSceneImages]:
        obs = self._reset()
        return self.from_internal(obs)

    def step(self, action: np.ndarray) -> tuple[TDScene, TDSceneImages]:
        obs = self._step(action)
        return self.from_internal(obs)

    @abc.abstractmethod
    def get_cursor(self, obs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def sample(self) -> tuple[TDScene, TDSceneImages]:
        scene, vision = self.reset()
        while self.is_bad_sample(scene):
            scene, vision = self.reset()
        return scene, vision

    def is_bad_sample(self, obs: TDScene) -> bool:
        return False  # By default, no sample is bad. Override in specific scenes if needed.

    @property
    @abc.abstractmethod
    def entities(self) -> list[Entity]:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def cursor(self) -> Entity:
        raise NotImplementedError()

    @abc.abstractmethod
    def heca_td(self, obs) -> TDScene:
        raise NotImplementedError()

    @abc.abstractmethod
    def image_tensors(self, obs) -> dict[str, torch.Tensor]:
        raise NotImplementedError()

    @abc.abstractmethod
    def to_td_scene_vision(self, obs) -> TDSceneImages:
        raise NotImplementedError()

    @abc.abstractmethod
    def image_numpy(self, obs) -> dict[str, np.ndarray]:
        raise NotImplementedError()

    @abc.abstractmethod
    def _reset(self) -> Any:
        raise NotImplementedError()

    @abc.abstractmethod
    def _step(self, action: np.ndarray) -> Any:
        raise NotImplementedError()

    def sample_image(self) -> dict[str, np.ndarray]:
        obs = self._reset()
        return self.image_numpy(obs)

    def _load(self, path: Path):
        dc_pattern = re.compile(rf"xk(\d+)_yk(\d+)_xs(\d+)_ys(\d+)\.png")
        sample_postfix = r"_sample(\d+)\.png"
        for kpt in self.kp_tuples:
            edir = path / kpt.cam / kpt.kp.cfg.label
            for state in kpt.kp.cfg.states:
                state_pattern = re.compile(rf"{re.escape(state)}{sample_postfix}")
                for file in edir.glob(f"{state}_sample*.png"):
                    if state_pattern.fullmatch(file.name):
                        self.state_samples[kpt][state].append(
                            Image.open(file),
                        )
            files = list(edir.glob(f"xk*_yk*_xs*_ys*.png"))
            assert files is not None
            assert len(files) == 1
            file = files[0]
            match = dc_pattern.fullmatch(file.name)
            if match:
                self.kp_samples[kpt] = (
                    Image.open(file),
                    int(match.group(1)),
                    int(match.group(2)),
                    int(match.group(3)),
                    int(match.group(4)),
                )

            dc_sample = self.kp_samples.get(kpt)
            state_samples = self.state_samples.get(kpt)

            if dc_sample is None or state_samples is None:
                logger.warning(
                    f"Missing samples for {kpt.cam} and {kpt.kp.cfg.label}. Skipping."
                )
                continue

            expected_states = set(kpt.kp.cfg.states)
            loaded_states = set(state_samples.keys())
            if loaded_states != expected_states:
                logger.warning(
                    f"State label mismatch for {kpt.cam} and {kpt.kp.cfg.label}. "
                    f"Expected {expected_states}, got {loaded_states}. Skipping."
                )
                continue

            self.prepare_references(
                entities=[kpt.kp],
                kp_references=[dc_sample],
                state_references=[state_samples],
            )

    def _save(self, path: Path):
        raise NotImplementedError()
