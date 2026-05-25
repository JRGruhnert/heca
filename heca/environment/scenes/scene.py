import abc
import re
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch
from PIL import Image, ImageTk
from tensordict import TensorDict

from heca.classes.persist import Persistable
from heca.entities.entity import Entity, Mobility
from heca.environment.scenes.image_extractor import ImageExtractor
from heca.misc import logger
from heca.misc.td import (
    TDEntities,
    TDEntity,
    TDImage,
    TDScene,
)


class KPTuple(NamedTuple):
    cam: str
    kp: Entity


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
            mobility=Mobility.FREE,
        )
        self.extractors = {
            name: ImageExtractor.create(extractor)
            for name, extractor in cfg.extractors.items()
        }

        # Kp selection
        self.kp_tuples: list[KPTuple] = []
        for cam in self.extractors.keys():
            for entity in self.entities:
                self.kp_tuples.append(
                    KPTuple(
                        cam=cam,
                        kp=entity,
                    ),
                )

        self.line_marker: int | None = None
        self.pred_marker: int | None = None
        self.manual_marker: int | None = None
        self.selection: KPTuple | None = None
        self.point: tuple[int, int] | None = None
        self.dc_values: tuple[Image.Image, int, int] | None = None
        self.title: str = "Selection for: {cam}.{kp}.{selection}"
        self.state_samples: dict[KPTuple, dict[str, list[Image.Image]]] = {
            triple: {state: [] for state in triple.kp.cfg.states}
            for triple in self.kp_tuples
        }
        self.kp_samples: dict[KPTuple, tuple[Image.Image, int, int, int, int]] = {}

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
        dc_postfix = r"_x1\d+_y1\d+_x2\d+_y2\d+\.png"
        sample_postfix = r"_sample\d+\.png"
        for kpt in self.kp_tuples:
            edir = path / kpt.cam / kpt.kp.cfg.label
            for state in kpt.kp.cfg.states:
                files.extend(list(edir.glob(f"{state}_sample*.png")))
            files.extend(list(edir.glob(f"{self.cfg.dc_label}_x1*_y1*_x2*_y2*.png")))

        for kpt in self.kp_tuples:
            for state in kpt.kp.cfg.states:
                pattern = re.compile(rf"({re.escape(state)}){sample_postfix}")
                rem = [(pattern.match(file.name), file) for file in files]
                rem = [(m, f) for m, f in rem if m is not None]
                for match, file in rem:
                    self.state_samples[kpt][state].append(
                        Image.open(file),
                    )
            pattern = re.compile(rf"{re.escape(self.cfg.dc_label)}{dc_postfix}")
            rem = [(pattern.match(file.name), file) for file in files]
            rem = [(m, f) for m, f in rem if m is not None]
            for match, file in rem:
                self.kp_samples[kpt] = (
                    Image.open(file),
                    int(match.group(1)),
                    int(match.group(2)),
                    int(match.group(3)),
                    int(match.group(4)),
                )

        for kpt in self.kp_tuples:
            for state in kpt.kp.cfg.states:
                state_samples = self.state_samples.get(kpt, None)
                dc_sample = self.kp_samples.get(kpt, None)
                # logger.debug(f'Found {len(rem)} matches for "{cam}" and "{e_label}"')
                # logger.debug(f"Matches: {rem.keys()}")
                # logger.debug(f"DC match: {dc_match}")
                if dc_sample is None:
                    logger.warning(
                        f"Missing DC samples for {kpt.cam} and {kpt.kp.cfg.label}. Skipping entity."
                    )
                    continue
                if state_samples is None or state_samples.keys() != set(
                    kpt.kp.cfg.states
                ):
                    logger.warning(
                        f"Missing state samples for {kpt.cam} and {kpt.kp.cfg.label}. Skipping entity."
                    )
                    continue
                self.extractors[kpt.cam].add_entity_sample_for_cam(
                    kpt.kp, dc_sample, state_samples
                )

    def _save(self, path: Path):
        for kpt, state_dict in self.state_samples.items():
            entity_dir = path / kpt.cam / kpt.kp.cfg.label
            entity_dir.mkdir(parents=True, exist_ok=True)
            for state, samples in state_dict.items():
                for idx, img in enumerate(samples):
                    img.save(
                        entity_dir / f"{state}_sample{idx}.png"
                    )  # e.g., "open_sample0.png"
            kps = self.kp_samples.get(kpt, None)
            if kps is not None:
                img, x1, y1, x2, y2 = kps
                img.save(
                    entity_dir / f"{self.cfg.dc_label}_x1{x1}_y1{y1}_x2{x2}_y2{y2}.png"
                )

    def make_rnd_image(self):
        assert self.selection is not None
        assert self.label is not None

        obs = self._reset()
        # num_actions = 10  # TODO: Tune this number based on the environment dynamics
        # for _ in range(num_actions):
        #    pos = torch.zeros(3)
        #    quat = torch.tensor([0, 0, 0, 1])
        #    gripper = torch.tensor(0.0)
        #    action = torch.cat([pos, quat, gripper]).numpy()
        #    obs = self._step(action)
        np_images_dict = self.image_numpy(obs)
        picture = np_images_dict[self.selection.cam]
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
        if self.dc_values is not None:
            self.place_predicted_marker(
                self.img, self.dc_values[0], self.dc_values[1], self.dc_values[2]
            )

    def place_predicted_marker(
        self, image: Image.Image, ref: Image.Image, u: int, v: int
    ):
        assert self.selection is not None
        extractor = self.extractors[self.selection.cam]
        x, y = extractor.encode_direct(image=image, ref_image=ref, x=u, y=v)
        x, y = self.scale_point_to_canvas(int(x), int(y))
        self.place_marker(x, y, "purple", self.pred_marker)

    def on_click(self, event: tk.Event):
        assert self.selection is not None
        self.point = self.scale_point_to_image(event.x, event.y)
        self.place_marker(event.x, event.y, "red", self.manual_marker)

        if self.label == self.cfg.dc_label and self.dc_values is None:
            # First click on new kp selection
            self.dc_values = (self.img, self.point[0], self.point[1])
            self.place_predicted_marker(
                self.img, self.dc_values[0], self.dc_values[1], self.dc_values[2]
            )
            if self.manual_marker is not None:
                self.canvas.delete(self.manual_marker)
        elif self.dc_values is not None:
            # Second click on same selection, update DC values
            self.kp_samples[self.selection] = (
                self.img,
                self.dc_values[1],
                self.dc_values[2],
                self.point[0],
                self.point[1],
            )
        else:
            raise NotImplementedError(
                f"Unexpected click state: label={self.label}, dc_values={self.dc_values}"
            )

    def scale_point_to_canvas(self, x: int, y: int) -> tuple[int, int]:
        # Scale the point according to the display size
        scaled_x = int(x * self.scale) + self.offset_x
        scaled_y = int(y * self.scale) + self.offset_y
        return scaled_x, scaled_y

    def scale_point_to_image(self, x: int, y: int) -> tuple[int, int]:
        # Remove the offset and scale back to original image coordinates
        real_x = int((x - self.offset_x) / self.scale)
        real_y = int((y - self.offset_y) / self.scale)
        return real_x, real_y

    def place_marker(self, x: int, y: int, color: str, marker: int | None):
        if marker is not None:
            self.canvas.delete(marker)

        # Draw a marker at the specified point
        self.manual_marker = self.canvas.create_oval(
            x - self.cfg.marker_radius,
            y - self.cfg.marker_radius,
            x + self.cfg.marker_radius,
            y + self.cfg.marker_radius,
            fill=color,
        )

    def add_image(self):
        assert self.selection is not None
        assert self.label is not None
        self.state_samples[self.selection][self.label].append(self.img)
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

        self.entity_labels: list[str] = []
        self.selection_iter = iter(self.kp_tuples)
        self.label_iter = iter(self.entity_labels)
        self.load_next_or_finish()
        self.window.mainloop()

    def load_next_or_finish(self):
        self.label = next(self.label_iter, None)
        if self.label is None:
            self.selection = next(self.selection_iter, None)
            if self.selection is None:
                self.window.quit()
                return
            else:
                self.label_iter = iter(self.selection.kp.cfg.states)
                self.label = self.cfg.dc_label
                self.dc_values = None
        assert self.label is not None
        assert self.selection is not None
        title = self.title.format(
            cam=self.selection.cam,
            kp=self.selection.kp.cfg.label,
            selection=self.label,
        )
        self.window.title(title)
        self.make_rnd_image()
