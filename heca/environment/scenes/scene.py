from dataclasses import dataclass
import abc
from tensordict import TensorDict
from PIL import Image, ImageTk
from pathlib import Path
from typing import Any, NamedTuple
import tkinter as tk
import numpy as np
import torch
import re

from heca.classes.persist import Persistable
from heca.entities.entity import Entity
from heca.environment.scenes.image_extractor import ImageExtractor
from heca.misc import logger
from heca.misc.td import (
    TDEntities,
    TDEntity,
    TDImage,
    TDScene,
)


class SelectionTriple(NamedTuple):
    cam_label: str
    entity_label: str
    selection_label: str


class Scene(Persistable):
    @dataclass(frozen=True, kw_only=True)
    class Location(Persistable.Location):
        folder: str = "samples"

    @dataclass(kw_only=True)
    class Config(Persistable.Config):
        gt: bool = False

        dc_label: str = "dc_pose"
        marker_radius: int = 3
        extractors: dict[str, ImageExtractor.Config]
        vis_size: tuple[int, int] = (512, 512)

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cursor = Entity.Config(
            label="cursor",
            states={"open", "closed"},
        )
        self.extractors = {
            name: ImageExtractor.create(extractor)
            for name, extractor in cfg.extractors.items()
        }

        # Kp selection
        self.triples: list[SelectionTriple] = []
        for cam in self.extractors.keys():
            for entity in self.entities:
                self.triples.append(
                    SelectionTriple(
                        cam_label=cam,
                        entity_label=entity.cfg.label,
                        selection_label=self.cfg.dc_label,
                    ),
                )
                for state_label in entity.cfg.states:
                    self.triples.append(
                        SelectionTriple(
                            cam_label=cam,
                            entity_label=entity.cfg.label,
                            selection_label=state_label,
                        ),
                    )

        self.triple = None
        self.point = None
        self.marker = None
        self.title = "Selection for: {cam}.{entity}.{selection}"
        self.loaded_samples: dict[
            SelectionTriple, list[tuple[Image.Image, int, int]]
        ] = {triple: [] for triple in self.triples}

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
            image_dict = self.to_td_image_dict(obs)
            pos, rot, state = self.get_cursor(obs)
            extracted = self.stitch_together(image_dict, pos, rot, state)
            data = {
                "tapas": self.tapas_td(obs, extracted),
                "heca": extracted,
            }
        return TDScene(data)

    @abc.abstractmethod
    def get_cursor(self, obs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def sample(self) -> TDScene:
        x = self.reset()
        while self.is_bad_sample(x):
            x = self.reset()
        return x

    def is_bad_sample(self, obs: TDScene) -> bool:
        return False  # By default, no sample is bad. Override in specific scenes if needed.

    @property
    @abc.abstractmethod
    def entities(self) -> list[Entity]:
        raise NotImplementedError()

    @abc.abstractmethod
    def tapas_td(self, obs, extracted: TDEntities | None = None) -> TensorDict:
        raise NotImplementedError()

    @abc.abstractmethod
    def heca_td(self, obs) -> TDEntities:
        raise NotImplementedError()

    @abc.abstractmethod
    def image_tensors(self, obs) -> dict[str, torch.Tensor]:
        raise NotImplementedError()

    @abc.abstractmethod
    def to_td_image_dict(self, obs) -> dict[str, TDImage]:
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

    def _relative_quaternion(
        self, q: torch.Tensor, q_ref: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the relative quaternion q_rel such that: q = q_rel * q_ref
        Returns q_rel = q * q_ref_conj
        """

        # q, q_ref: (..., 4) in (w, x, y, z) or (x, y, z, w) format
        # Assume (x, y, z, w) format as is common in PyTorch/robotics
        # Convert to (w, x, y, z) for computation if needed
        # Here, we assume (x, y, z, w)
        def quat_conj(q):
            return torch.tensor(
                [-q[0], -q[1], -q[2], q[3]], device=q.device, dtype=q.dtype
            )

        def quat_mult(q1, q2):
            x1, y1, z1, w1 = q1
            x2, y2, z2, w2 = q2
            w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
            x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
            y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
            z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
            return torch.stack([x, y, z, w])

        q_ref_conj = quat_conj(q_ref)
        q_rel = quat_mult(q, q_ref_conj)
        return q_rel

    def stitch_together(
        self,
        td_image_dict: dict[str, TDImage],
        cursor_pos: torch.Tensor,
        cursor_rot: torch.Tensor,
        cursor_state: torch.Tensor,
    ) -> TDEntities:
        all_kps = []
        all_states = []
        all_infos = []
        all_masks = []
        all_scores = []

        for td_image in td_image_dict.values():
            kps, kps_mask, states, scores, info = self.extractors[""].encode(td_image)
            all_kps.append(kps)
            all_states.append(states)
            all_infos.append(info)
            all_masks.append(kps_mask)
            all_scores.append(scores)

        kps_stack = torch.stack(all_kps)  # (C, K, D)
        kps_mask = torch.stack(all_masks)  # (C, K)
        scores = torch.stack(all_scores)  # (C, K)
        states_stack = torch.stack(all_states)  # (C, K, S)

        # Aggregate keypoints of different cameras
        num_kps = kps_stack.shape[1]
        final_kps = []
        final_states = []
        for k in range(num_kps):
            present = kps_mask[:, k] > 0
            if present.sum() == 0:
                # Not present in any camera
                final_kps.append(torch.full_like(kps_stack[0, k], float("nan")))
                final_states.append(torch.full_like(states_stack[0, k], float("nan")))
            elif present.sum() == 1:
                # Present in only one camera
                idx = present.nonzero(as_tuple=True)[0][0]
                final_kps.append(kps_stack[idx, k])
                final_states.append(states_stack[idx, k])
            else:
                # Present in multiple cameras
                # Mean for keypoints
                # Best score for states
                vals = kps_stack[present, k]
                final_kps.append(vals.mean(dim=0))
                valid_scores = scores[:, k].clone()
                valid_scores[~present] = float("-inf")
                idx = torch.argmax(valid_scores)
                final_states.append(states_stack[idx, k])
        final_kps = torch.stack(final_kps)  # (K, D)
        final_states = torch.stack(final_states)  # (K, S)

        td_entities: dict[str, TDEntity] = {}
        # Find the index of the end effector (ee) entity in self.entities
        num_present = min(final_kps.shape[0], final_states.shape[0], len(self.entities))
        for idx, entity in enumerate(self.entities):
            if idx >= num_present:
                # No keypoint or descriptor for this entity, skip
                # Assuming keypoints are ordered the same as entities
                continue

            tg_kp = final_kps[idx]
            rel_pos = tg_kp[:3] - cursor_pos
            rel_rot = self._relative_quaternion(
                tg_kp[3:7],
                cursor_rot,
            )
            td_rel = TDEntity(
                position=rel_pos,
                rotation=rel_rot,
                state=final_states[idx],
            )
            td = TDEntity(
                position=tg_kp[:3],
                rotation=tg_kp[3:7],
                state=final_states[idx],
            )
            td_entities[f"{entity.cfg.label}_rel"] = td_rel
            td_entities[entity.cfg.label] = td
        td_entities["cursor"] = TDEntity(
            position=cursor_pos,
            rotation=cursor_rot,
            state=cursor_state,
        )
        return TDEntities(td_entities)

    def _load(self, path: Path):
        files: list[Path] = []
        postfix = r"_ex\d+_x-?\d+(?:\.\d+)?_y-?\d+(?:\.\d+)?\.png"
        for triple in self.triples:
            entity_dir = path / triple.cam_label / triple.entity_label
            files.extend(
                list(entity_dir.glob(f"{triple.selection_label}_ex*_x*_y*.png"))
            )
        for triple in self.triples:
            pattern = re.compile(rf"({re.escape(triple.selection_label)}){postfix}")
            rem = [(pattern.match(file.name), file) for file in files]
            rem = [(m, f) for m, f in rem if m is not None]
            for match, file in rem:
                # selection_label = match.group(1)
                # ex_num = int(match.group(2))
                x_val = int(match.group(3))
                y_val = int(match.group(4))
                # Load the image and store it in the appropriate extractor
                img = Image.open(file)
                self.loaded_samples[triple].append((img, x_val, y_val))

        for cam, extractor in self.extractors.items():
            for entity in self.entities:
                e_label = entity.cfg.label
                rem = {
                    key: value
                    for key, value in self.loaded_samples.items()
                    if key.cam_label == cam and key.entity_label == e_label
                }
                dc_match = next(
                    (
                        value
                        for key, value in rem.items()
                        if key.selection_label == self.cfg.dc_label
                    ),
                    None,
                )
                assert dc_match is not None
                # logger.debug(f'Found {len(rem)} matches for "{cam}" and "{e_label}"')
                # logger.debug(f"Matches: {rem.keys()}")
                # logger.debug(f"DC match: {dc_match}")
                state_matches = {
                    key.selection_label: value
                    for key, value in rem.items()
                    if key.selection_label in entity.cfg.states
                }
                assert state_matches.keys() == set(entity.cfg.states)
                if len(dc_match) == 0:
                    logger.warning(
                        f"Missing DC samples for {cam} and {e_label}. Skipping entity."
                    )
                    continue
                if any(len(matches) == 0 for matches in state_matches.values()):
                    logger.warning(
                        f"Missing state samples for {cam} and {e_label}. Skipping entity."
                    )
                    continue
                extractor.add_entity_sample_for_cam(entity, dc_match, state_matches)

    def _save(self, path: Path):
        for triple, samples in self.loaded_samples.items():
            entity_dir = path / triple.cam_label / triple.entity_label
            entity_dir.mkdir(parents=True, exist_ok=True)
            for idx, (img, x_val, y_val) in enumerate(samples):
                img.save(
                    entity_dir
                    / f"{triple.selection_label}_ex{idx}_x{x_val}_y{y_val}.png"
                )

    def make_rnd_image(self):
        assert self.triple is not None
        obs = self._reset()
        # num_actions = 10  # TODO: Tune this number based on the environment dynamics
        # for _ in range(num_actions):
        #    pos = torch.zeros(3)
        #    quat = torch.tensor([0, 0, 0, 1])
        #    gripper = torch.tensor(0.0)
        #    action = torch.cat([pos, quat, gripper]).numpy()
        #    obs = self._step(action)
        np_images_dict = self.image_numpy(obs)
        picture = np_images_dict[self.triple[0]]
        self.img = Image.fromarray(picture)

        orig_w, orig_h = self.img.size

        self.scale = min(
            self.cfg.vis_size[0] / orig_w,
            self.cfg.vis_size[1] / orig_h,
        )
        display_w = int(orig_w * self.scale)
        display_h = int(orig_h * self.scale)

        self.display_img = self.img.resize(
            (display_w, display_h),
            Image.Resampling.NEAREST,
        )
        self.img_tk = ImageTk.PhotoImage(self.display_img)
        self.canvas.delete("all")

        self.offset_x = (self.cfg.vis_size[0] - display_w) // 2
        self.offset_y = (self.cfg.vis_size[1] - display_h) // 2
        self.canvas.create_image(
            self.offset_x,
            self.offset_y,
            anchor="nw",
            image=self.img_tk,
        )

    def on_click(self, event: tk.Event):
        # remove canvas offset
        x = event.x - self.offset_x
        y = event.y - self.offset_y

        # convert back to original image coordinates
        real_x = int(x / self.scale)
        real_y = int(y / self.scale)

        self.point = (real_x, real_y)

        if self.marker:
            self.canvas.delete(self.marker)

        # Draw a marker at the clicked point
        self.marker = self.canvas.create_oval(
            event.x - self.cfg.marker_radius,
            event.y - self.cfg.marker_radius,
            event.x + self.cfg.marker_radius,
            event.y + self.cfg.marker_radius,
            fill="red",
        )

    def add_image(self):
        assert self.point is not None
        assert self.triple is not None

        self.loaded_samples[self.triple].append(
            (self.img, self.point[0], self.point[1])
        )
        self.make_rnd_image()

    def sample_selection(self):
        self.window = tk.Tk()
        self.canvas = tk.Canvas(
            self.window,
            width=self.cfg.vis_size[0],
            height=self.cfg.vis_size[1],
        )
        self.canvas.pack()
        btn_frame = tk.Frame(self.window)
        btn_frame.pack(side="top")
        # controls
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.pack()
        tk.Button(btn_frame, text="Add", bg="green", command=self.add_image).pack(
            side="left"
        )
        tk.Button(
            btn_frame, text="Next", bg="red", command=self.load_next_or_finish
        ).pack(side="left")
        tk.Button(
            btn_frame, text="Resample", bg="blue", command=self.make_rnd_image
        ).pack(side="left")

        self.triples_iter = iter(self.triples)
        self.load_next_or_finish()
        self.window.mainloop()

    def load_next_or_finish(self):
        self.triple = next(self.triples_iter, None)
        if self.triple is None:
            self.window.quit()
            return
        title = self.title.format(
            cam=self.triple.cam_label,
            entity=self.triple.entity_label,
            selection=self.triple.selection_label,
        )
        self.window.title(title)
        self.make_rnd_image()
