import abc
import json
import os
import time
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


def reference_note(reference: Reference) -> dict[str, Any]:
    """One reference as the ``annotation.json`` entry ``Scene._save`` writes."""
    note: dict[str, Any] = {
        "x": int(reference.x),
        "y": int(reference.y),
        "xyz": [float(v) for v in reference.xyz],
    }
    if reference.frame is not None:
        note["frame"] = int(reference.frame)
    return note


def reference_from_note(note: dict, image: Image.Image) -> Reference:
    """Read back what :func:`reference_note` wrote."""
    frame = note.get("frame")
    return Reference(
        image=image,
        x=int(note["x"]),
        y=int(note["y"]),
        xyz=np.asarray(note["xyz"], dtype=np.float64),
        frame=None if frame is None else int(frame),
    )


def write_atomically(path: Path, write: Any) -> None:
    """
    Replace ``path`` only once it is complete.

    Several processes read the scene (every ``Heca`` loads it, every ``ExpertModel``
    reads the reference images) while every ``Heca`` in the same start-up writes it,
    so an in-place writer can be read half-finished: an ``Image.open`` that lands
    inside the truncate-then-write window of ``Image.save`` raises
    ``UnidentifiedImageError``, and a ``json.loads`` there raises ``JSONDecodeError``.
    Writing to a private file and renaming it hides the writes instead of racing them.
    """
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def load_reference_image(path: Path, attempts: int = 5) -> Image.Image:
    """Read one reference image completely, so a lazy handle cannot go stale.

    Loads the pixels while the file is known to be readable, which both validates it
    and frees the file handle that a later in-place ``Image.save`` on this path would
    otherwise read from. Retries cover a writer that predates the atomic save.
    """
    for attempt in range(attempts):
        try:
            image = Image.open(path)
            image.load()
            return image
        except (OSError, ValueError) as error:
            if attempt == attempts - 1:
                raise OSError(
                    f"cannot read the reference image {path}: {error}. Re-extract it "
                    f"from the data archive (the click itself is in "
                    f"{path.with_name('annotation.json')})"
                ) from error
            time.sleep(0.1 * (attempt + 1))
    raise AssertionError("unreachable")


def save_reference(
    edir: Path, name: str, reference: Reference, note: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Write one reference beside its siblings, returning its annotation entry.

    Unchanged references are left alone: rewriting them is what made a concurrent
    scene load fail, and it is the common case (the scene is saved on every start).
    """
    edir.mkdir(parents=True, exist_ok=True)
    written = reference_note(reference)
    target = edir / f"{name}.png"
    if target.exists() and note == written:
        return written
    write_atomically(target, lambda p: reference.image.save(p, format="PNG"))
    return written


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
        self.references: dict[str, dict[str, Reference]] = {}
        self.extra_ranges: dict[str, tuple[float, float]] = {}
        self.extra_references: dict[str, dict[str, np.ndarray]] = {}
        self.extra_positions: dict[str, list[np.ndarray]] = {}
        self.render_frames = False

    def add_extra_range(self, label: str, lo: float, hi: float) -> None:
        """Fold one expert's demo range into the scene's, over all its experts."""
        old = self.extra_ranges.get(label)
        self.extra_ranges[label] = (
            (lo, hi) if old is None else (min(old[0], lo), max(old[1], hi))
        )

    def add_extra_positions(self, label: str, positions: np.ndarray) -> None:
        self.extra_positions.setdefault(label, []).append(
            np.atleast_2d(np.asarray(positions))
        )

    def finish_extras(self) -> None:
        """Derive each label's references from every expert's encoded positions."""
        for label, chunks in self.extra_positions.items():
            found = self.entities[label].reference_positions(
                np.concatenate(chunks, axis=0)
            )
            self.extra_references[label] = found if found is not None else {}
        self.extra_positions = {}

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

    def sample_task(self, with_scenes: bool = True) -> tuple[
        tuple[DCScene, TDImage],
        tuple[DCScene, TDImage],
    ]:
        self.current_step = 0
        return self._sample_task(with_scenes)

    def _sample_task(self, with_scenes: bool = True) -> tuple[
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
        with_scenes: bool = True,
    ) -> tuple[list[list[DCScene]], list[list[TDImage]]]:
        raise NotImplementedError()

    def _load(self, path: Path) -> bool:
        self.references = {}
        self.extra_references = {}
        self.extra_positions = {}
        for label, entity in self.entities.items():
            edir = path / label
            self.references[label] = {}
            stored = edir / "annotation.json"
            notes = json.loads(stored.read_text()) if stored.exists() else {}
            for name in (Reference.POSITION, *entity.reference_names):
                note = notes.get(name)
                if note is None:
                    continue
                self.references[label][name] = reference_from_note(
                    note, load_reference_image(edir / f"{name}.png")
                )
            extra = notes.get("extra")
            if extra is not None:
                self.extra_ranges[label] = (float(extra[0]), float(extra[1]))
            derived = notes.get("extra_references")
            if derived is not None:
                self.extra_references[label] = {
                    name: np.asarray(xyz, dtype=np.float64)
                    for name, xyz in derived.items()
                }
        return True

    def _save(self, path: Path) -> bool:
        for label, entity in self.entities.items():
            edir = path / label
            edir.mkdir(parents=True, exist_ok=True)
            existing = edir / "annotation.json"
            notes: dict[str, dict[str, Any]] = (
                json.loads(existing.read_text()) if existing.exists() else {}
            )
            for name, reference in self.references.get(label, {}).items():
                notes[name] = save_reference(edir, name, reference, notes.get(name))
            if label in self.extra_ranges:
                notes["extra"] = list(self.extra_ranges[label])
            if label in self.extra_references:
                notes["extra_references"] = {
                    name: [float(v) for v in xyz]
                    for name, xyz in self.extra_references[label].items()
                }
            text = json.dumps(notes, indent=2) + "\n"
            stored = edir / "annotation.json"
            if not stored.exists() or stored.read_text() != text:
                write_atomically(stored, lambda p: p.write_text(text))
        return True

    def references_ready(self) -> bool:
        """Every reference the scene's entities need is present."""
        for label, entity in self.entities.items():
            have = self.references.get(label, {})
            for name in (Reference.POSITION, *entity.reference_names):
                if name not in have:
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
