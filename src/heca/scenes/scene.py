from collections import defaultdict
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

    def from_internal(self, data) -> tuple[DCScene, TDImage, np.ndarray]:
        tdscene = self.to_dc_scene(data)
        tdimage = self.to_td_image(data)
        npimage = self.to_np_image(data)
        return tdscene, tdimage, npimage

    def step(self, action: np.ndarray) -> tuple[DCScene, TDImage, float, bool, bool]:
        obs, reward, terminal, truncated = self._step(action)
        tdscene, tdimage, _ = self.from_internal(obs)
        return tdscene, tdimage, reward, terminal, truncated

    def step_vis(self, action: np.ndarray) -> tuple[DCScene, TDImage, np.ndarray]:
        obs, _, _, _ = self._step(action)
        return self.from_internal(obs)

    @abc.abstractmethod
    def _step(self, action: np.ndarray) -> tuple[Any, float, bool, bool]:
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

    @abc.abstractmethod
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
    ) -> tuple[list[list[DCScene]], list[list[TDImage]]]:
        raise NotImplementedError()

    def _load(self, path: Path, tag: str):
        dc_pattern = re.compile(rf"xk(\d+)_yk(\d+)_xs(\d+)_ys(\d+)\.png")
        sample_postfix = r"_sample(\d+)\.png"
        for label, entity in self.entities.items():
            edir = path / tag / label
            self.state_references[label] = {}
            for state in entity.cfg.states:
                self.state_references[label][state] = []
                state_pattern = re.compile(rf"{re.escape(state)}{sample_postfix}")
                for file in edir.glob(f"{state}_sample*.png"):
                    if state_pattern.fullmatch(file.name):
                        self.state_references[label][state].append(
                            Image.open(file),
                        )
            files = list(edir.glob(f"xk*_yk*_xs*_ys*.png"))
            assert files is not None
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

    def _save(self, path: Path, tag: str):
        for label in self.entities.keys():
            entity_dir = path / tag / label
            entity_dir.mkdir(parents=True, exist_ok=True)
            for state, samples in self.state_references[label].items():
                for idx, img in enumerate(samples):
                    img.save(entity_dir / f"{state}_sample{idx}.png")
            img, x1, y1, x2, y2 = self.kp_references[label]
            img.save(entity_dir / f"xk{x1}_yk{y1}_xs{x2}_ys{y2}.png")

    @property
    def description(self) -> str:
        raise NotImplementedError()

    @property
    def entities(self) -> dict[str, Entity]:
        raise NotImplementedError()

    def demo_auto_extract(self):
        with h5py.File(self.dataset_path, "r") as f:
            done = np.asarray(f["oracle_done"])[:, 0]
            success = np.asarray(f["oracle_success"])[:, 0]
            task = np.asarray(f["privileged_target_task"], dtype=str)

            # Find segment boundaries (indices where oracle switches)
            done_idxs = np.where(done == 1.0)[0]
            seg_starts = [0] + list(done_idxs)
            seg_ends = list(done_idxs) + [len(done)]

            agent_data = defaultdict(list)

            for start, end in zip(seg_starts, seg_ends):
                if start >= end:
                    continue
                if success[end - 1] != 1.0:
                    continue

                agent_key = self.entities[task[end - 1]].agent_key(f, start, end - 1)
                agent_data[agent_key].append((start, end - 1))

            # Write each bucket to a separate HDF5 file
            all_keys = list(f.keys())
            for agent_key, segments in agent_data.items():
                out_path = self.output_dir / f"{agent_key}.h5"
                with h5py.File(out_path, "w") as out:
                    demo_id = 0
                    for start, end in segments:
                        demo_ids = np.full(end - start + 1, demo_id, dtype=np.int32)
                        for key in all_keys:
                            data = f[key][start : end + 1]
                            if "demo" not in out and key == "demo":
                                continue  # skip if demo dataset doesn't exist yet
                            if key not in out:
                                maxshape = (None,) + data.shape[1:]
                                out.create_dataset(
                                    key,
                                    data=data,
                                    maxshape=maxshape,
                                    chunks=(1,) + data.shape[1:],
                                )
                            else:
                                ds = out[key]
                                n = ds.shape[0]
                                ds.resize(n + len(data), axis=0)
                                ds[n : n + len(data)] = data
                        # Add demo_id dataset
                        if "demo" not in out:
                            out.create_dataset(
                                "demo", data=demo_ids[:1], maxshape=(None,), chunks=(1,)
                            )
                        else:
                            ds = out["demo"]
                            n = ds.shape[0]
                            ds.resize(n + len(demo_ids), axis=0)
                            ds[n : n + len(demo_ids)] = demo_ids
                        demo_id += 1

                print(f"  {agent_key}: {len(segments)} demos → {out_path}")
