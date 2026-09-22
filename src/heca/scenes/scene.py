import abc
import json
import h5py
import torch
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from PIL import Image


from heca.data.data import DCScene, TDImage
from heca.data.entity import Entity
from heca.data.reference import Reference
from heca.misc.base import Persistable


@dataclass(kw_only=True, slots=True)
class SceneFeedback:
    terminal: bool
    reward: float
    truncated: bool
    budget: float

    @property
    def success(self) -> bool:
        return self.terminal and self.reward > 0.5

    @property
    def end(self) -> bool:
        return self.truncated or self.terminal


class Scene(Persistable):
    @dataclass(kw_only=True)
    class Config(Persistable.Config):
        folder: str = "scenes"
        width: int = 256
        height: int = 256
        viewer: bool = False
        gated_fail_prob: float = 0.2
        success_reward: float = 1.0
        step_reward: float = -0.01
        max_steps: int = 16

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.current_step = 0

        # label -> reference name -> annotation.  "pos" is the entity's own
        # reference picture, the names in ``Entity.reference_names`` are the joint
        # references its class needs.
        self.references: dict[str, dict[str, Reference]] = {}
        # label -> state index -> pictures, only for entities with more than one
        # state: a single state is constant, so there is nothing to look at.
        self.state_references: dict[str, dict[int, list[Image.Image]]] = {}

    def from_internal(self, data) -> tuple[DCScene, TDImage, np.ndarray]:
        tdscene = self.to_dc_scene(data)
        tdimage = self.to_td_image(data)
        npimage = self.to_np_image(data)
        return tdscene, tdimage, npimage

    def apply_truncation(self, lfb: SceneFeedback) -> SceneFeedback:
        """Reward shaping for one low-level action (no option counting)."""
        reward = self.cfg.step_reward + self.cfg.success_reward * int(lfb.success)
        return SceneFeedback(
            reward=reward,
            terminal=lfb.terminal,
            truncated=lfb.truncated,
            budget=self.budget,
        )

    def count_option(self, fb: SceneFeedback) -> SceneFeedback:
        self.current_step += 1
        truncated = self.current_step >= self.cfg.max_steps or fb.truncated
        return SceneFeedback(
            reward=fb.reward,
            terminal=fb.terminal,
            truncated=truncated,
            budget=self.budget,
        )

    @property
    def budget(self) -> float:
        return max(self.cfg.max_steps - self.current_step, 0) / self.cfg.max_steps

    def step(self, action: np.ndarray) -> tuple[DCScene, TDImage, SceneFeedback]:
        obs, fb = self._step(action)
        return self.package_internal(obs, fb)

    def step_virt(
        self,
        x: DCScene,
        y: DCScene,
        elabels: list[str],
        rand_noise: float = 0.0,
        fail_noise: float = 0.0,
    ) -> tuple[DCScene, TDImage, SceneFeedback]:
        obs, fb = self._step_virt(x, y, elabels, rand_noise, fail_noise)
        return self.package_internal(obs, fb)

    def gated_step_virt(self, x: DCScene) -> tuple[DCScene, SceneFeedback]:
        failed = bool(np.random.random() < self.cfg.gated_fail_prob)
        fb = SceneFeedback(
            terminal=failed,
            reward=0.0,  # shaped by apply_truncation below (no success reward)
            truncated=False,
            budget=self.budget,
        )
        return x, self.apply_truncation(fb)

    def package_internal(
        self, obs: dict, lfb: SceneFeedback
    ) -> tuple[DCScene, TDImage, SceneFeedback]:
        tdscene, tdimage, _ = self.from_internal(obs)
        fb = self.apply_truncation(lfb)
        return tdscene, tdimage, fb

    @abc.abstractmethod
    def _step(self, action: np.ndarray) -> tuple[Any, SceneFeedback]:
        raise NotImplementedError()

    @abc.abstractmethod
    def _step_virt(
        self,
        x: DCScene,
        y: DCScene,
        elabels: list[str],
        rand_noise: float,
        fail_noise: float,
    ) -> tuple[Any, SceneFeedback]:
        raise NotImplementedError()

    def sample_task(self) -> tuple[
        tuple[DCScene, TDImage],
        tuple[DCScene, TDImage],
    ]:
        self.current_step = 0
        return self._sample_task()

    def _sample_task(self) -> tuple[
        tuple[DCScene, TDImage],
        tuple[DCScene, TDImage],
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
        annotation = "annotation.json"
        self.references = {}
        self.state_references = {}
        for label, entity in self.entities.items():
            edir = path / label
            self.references[label] = {}
            self.state_references[label] = {}
            notes = json.loads((edir / annotation).read_text()) if (edir / annotation).exists() else {}
            for name in (Reference.POSITION, *entity.reference_names):
                note = notes.get(name)
                if note is None:
                    continue
                self.references[label][name] = Reference(
                    image=Image.open(edir / f"{name}.png"),
                    x=int(note["x"]),
                    y=int(note["y"]),
                    xyz=np.asarray(note["xyz"], dtype=np.float64),
                )
            if entity.cfg.n_states > 1:
                for idx in range(entity.cfg.n_states):
                    self.state_references[label][idx] = [
                        Image.open(file)
                        for file in sorted(edir.glob(f"state{idx}_sample*.png"))
                    ]
        return True

    def _save(self, path: Path) -> bool:
        for label, entity in self.entities.items():
            edir = path / label
            edir.mkdir(parents=True, exist_ok=True)
            notes: dict[str, dict[str, Any]] = {}
            for name, reference in self.references[label].items():
                reference.image.save(edir / f"{name}.png")
                notes[name] = {
                    "x": int(reference.x),
                    "y": int(reference.y),
                    "xyz": [float(v) for v in reference.xyz],
                }
            (edir / "annotation.json").write_text(json.dumps(notes, indent=2) + "\n")
            if entity.cfg.n_states > 1:
                for state, samples in self.state_references[label].items():
                    for idx, img in enumerate(samples):
                        img.save(edir / f"state{state}_sample{idx}.png")
        return True

    def references_ready(self) -> bool:
        """Every reference and state sample the scene's entities need is present."""
        for label, entity in self.entities.items():
            have = self.references.get(label, {})
            for name in (Reference.POSITION, *entity.reference_names):
                if name not in have:
                    return False
            if entity.cfg.n_states > 1:
                states = self.state_references.get(label, {})
                if any(not states.get(idx) for idx in range(entity.cfg.n_states)):
                    return False
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
