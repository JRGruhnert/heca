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
        # self.dc_values: tuple[Image.Image, int, int] | None = None
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
            kps, states, info = self.extractors[""].encode(td_image)
            all_kps.append(kps)
            all_states.append(states)
            all_infos.append(info)
            all_masks.append(info["kp_mask"])
            all_scores.append(info["state_scores"])

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
        dc_pattern = re.compile(
            rf"{re.escape(self.cfg.dc_label)}_xk(\d+)_yk(\d+)_xs(\d+)_ys(\d+)\.png"
        )
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
            for file in edir.glob(f"{self.cfg.dc_label}_xk*_yk*_xs*_ys*.png"):
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

            self.extractors[kpt.cam].add_entity_sample_for_cam(
                kpt.kp,
                dc_sample,
                state_samples,
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
            kps = self.kp_samples.get(kpt)
            if kps is not None:
                img, x1, y1, x2, y2 = kps
                img.save(
                    entity_dir / f"{self.cfg.dc_label}_xk{x1}_yk{y1}_xs{x2}_ys{y2}.png"
                )

    def delete_from_canvas(self, marker: str):
        if marker == "manual":
            if self.manual_marker is not None:
                self.canvas.delete(self.manual_marker)
            self.manual_marker = None
        elif marker == "pred":
            if self.pred_marker is not None:
                self.canvas.delete(self.pred_marker)
            self.pred_marker = None
        elif marker == "line":
            if self.line_marker is not None:
                self.canvas.delete(self.line_marker)
            self.line_marker = None
        else:
            logger.warning(f"Unknown marker type: {marker}")

    def update_line_marker(self):
        self.delete_from_canvas("line")

        if self.manual_marker is None or self.pred_marker is None:
            return

        mx1, my1, mx2, my2 = self.canvas.coords(self.manual_marker)
        px1, py1, px2, py2 = self.canvas.coords(self.pred_marker)

        mx = (mx1 + mx2) / 2.0
        my = (my1 + my2) / 2.0
        px = (px1 + px2) / 2.0
        py = (py1 + py2) / 2.0

        self.line_marker = self.canvas.create_line(
            mx, my, px, py, fill="yellow", width=2
        )

    def place_marker(self, x: int, y: int, color: str, marker: str, label: str = ""):
        ref = self.canvas.create_oval(
            x - self.cfg.marker_radius,
            y - self.cfg.marker_radius,
            x + self.cfg.marker_radius,
            y + self.cfg.marker_radius,
            fill=color,
        )
        # if label != "":
        #    self.canvas.create_text(
        #        x + self.cfg.marker_radius + 5,
        #        y - self.cfg.marker_radius,
        #        text=label,
        #        fill=color,
        #        anchor="w",
        #        font=("Arial", 8),
        #    )

        if marker == "manual":
            self.delete_from_canvas("manual")
            self.manual_marker = ref
        elif marker == "pred":
            self.delete_from_canvas("pred")
            self.pred_marker = ref
        else:
            logger.warning(f"Unknown marker type: {marker}")
            self.canvas.delete(ref)
            return

        self.update_line_marker()

    def make_rnd_image(self):
        self.pred_marker = None
        self.manual_marker = None
        self.line_marker = None
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
        self.offset_x = (self.cfg.vis_size[0] - display_w) // 2
        self.offset_y = (self.cfg.vis_size[1] - display_h) // 2

        self.display_img = self.img.resize(
            (display_w, display_h),
            Image.Resampling.NEAREST,
        )
        self.img_tk = ImageTk.PhotoImage(self.display_img)
        self.canvas.delete("all")

        self.canvas.create_image(
            self.offset_x,
            self.offset_y,
            anchor="nw",
            image=self.img_tk,
        )
        if not self.dc_selection:
            ref_img, dc_x, dc_y, sel_x, sel_y = self.kp_samples[self.selection]
            pred_dc_x, pred_dc_y, pred_canvas_x, pred_canvas_y = self.predict_dc_point(
                self.img,
                ref_img,
                dc_x,
                dc_y,
            )
            logger.info(
                f"Predicted DC point: ({pred_dc_x}, {pred_dc_y}), "
                f"Reference DC point: ({dc_x}, {dc_y}), "
                f"Selected point: ({sel_x}, {sel_y})"
            )
            rel_x, rel_y = self.make_relative_canvas_point(dc_x, dc_y, sel_x, sel_y)
            sel_canvas_x = pred_canvas_x - rel_x
            sel_canvas_y = pred_canvas_y - rel_y

            self.place_marker(pred_canvas_x, pred_canvas_y, "blue", "pred")
            self.place_marker(sel_canvas_x, sel_canvas_y, "red", "manual")

    def predict_dc_point(
        self, image: Image.Image, ref_image: Image.Image, ref_x: int, ref_y: int
    ) -> tuple[int, int, int, int]:
        assert self.selection is not None
        extractor = self.extractors[self.selection.cam]
        y, x = extractor.encode_direct(
            image=image, ref_image=ref_image, y=ref_y, x=ref_x
        )
        dc_x, dc_y = self.scale_point_to_canvas(int(x), int(y))
        return int(x), int(y), dc_x, dc_y

    def update_add_button_visibility(self):
        assert self.label is not None
        if self.label == self.cfg.dc_label:
            if self.add_btn.winfo_ismapped():
                self.add_btn.pack_forget()
        else:
            if not self.add_btn.winfo_ismapped():
                self.add_btn.pack(side="left")

    def make_relative_canvas_point(
        self, dc_x: int, dc_y: int, sel_x: int, sel_y: int
    ) -> tuple[int, int]:
        dc_x_canvas, dc_y_canvas = self.scale_point_to_canvas(dc_x, dc_y)
        sel_x_canvas, sel_y_canvas = self.scale_point_to_canvas(sel_x, sel_y)
        rel_x = dc_x_canvas - sel_x_canvas
        rel_y = dc_y_canvas - sel_y_canvas
        return rel_x, rel_y

    def place_relative_marker(self, x: int, y: int, rel_x: int, rel_y: int):
        abs_x = x + rel_x
        abs_y = y + rel_y
        self.place_marker(abs_x, abs_y, "red", "manual")

    def on_click(self, event: tk.Event):
        assert self.selection is not None
        self.point = self.scale_point_to_image(event.x, event.y)
        self.place_marker(event.x, event.y, "red", "manual")
        if self.label == self.cfg.dc_label:
            if self.dc_selection:
                self.dc_selection = False
                logger.debug(f"First Click point={self.point}")
                x, y = self.point
                self.kp_samples[self.selection] = (self.img, x, y, 0, 0)
                _, _, pred_canvas_x, pred_canvas_y = self.predict_dc_point(
                    self.img, self.img, x, y
                )
                self.place_marker(pred_canvas_x, pred_canvas_y, "blue", "pred")
                return
        logger.debug(f"Updated point={self.point} for selection {self.selection}")
        self.update_kp_sample_with_selection(self.point)

    def update_kp_sample_with_selection(self, point: tuple[int, int]):
        assert self.selection is not None
        if self.selection in self.kp_samples:
            img, dc_x, dc_y, _, _ = self.kp_samples[self.selection]
            x, y = point
            self.kp_samples[self.selection] = (img, dc_x, dc_y, x, y)

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

    def add_image(self):
        assert self.selection is not None
        assert self.label is not None

        if self.label == self.cfg.dc_label:
            logger.warning("Cannot add sample for DC label. Ignoring.")
            return

        self.state_samples[self.selection][self.label].append(self.img)
        self.make_rnd_image()

    def make_rnd_image2(self):
        cam = "front"
        obs = self._reset()
        td_dict = self.to_td_image_dict(obs)
        td_image = td_dict[cam]
        np_images_dict = self.image_numpy(obs)
        picture = np_images_dict[cam]
        self.img = Image.fromarray(picture)

        orig_w, orig_h = self.img.size

        self.scale = min(
            self.cfg.vis_size[0] / orig_w,
            self.cfg.vis_size[1] / orig_h,
        )
        display_w = int(orig_w * self.scale)
        display_h = int(orig_h * self.scale)
        self.offset_x = (self.cfg.vis_size[0] - display_w) // 2
        self.offset_y = (self.cfg.vis_size[1] - display_h) // 2

        self.display_img = self.img.resize(
            (display_w, display_h),
            Image.Resampling.NEAREST,
        )
        self.img_tk = ImageTk.PhotoImage(self.display_img)
        self.canvas.delete("all")

        self.canvas.create_image(
            self.offset_x,
            self.offset_y,
            anchor="nw",
            image=self.img_tk,
        )
        _, states, info = self.extractors[cam].encode(td_image)
        n_states = states.squeeze().tolist()
        y_pixel = info["y_pixel"].squeeze().tolist()
        x_pixel = info["x_pixel"].squeeze().tolist()
        logger.info(f"Predicted states: {n_states}, at pixels: ({x_pixel}, {y_pixel})")

        for idx, entity in enumerate(self.entities):
            if idx >= len(n_states):
                break
            state = n_states[idx]
            y_e = y_pixel[idx]
            x_e = x_pixel[idx]
            logger.info(
                f"Entity: {entity.cfg.label}, State: {state}, at pixels: ({x_e}, {y_e})"
            )
            y_canvas, x_canvas = self.scale_point_to_canvas(x_e, y_e)
            self.place_marker(x_canvas, y_canvas, "blue", "pred")

    def show_predictions(self):
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
        tk.Button(
            btn_frame, text="Sample", bg="#BFDBFE", command=self.make_rnd_image2
        ).pack(side="left")

        self.window.mainloop()

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
        self.add_btn = tk.Button(
            btn_frame, text="Add", bg="#D1FAE5", command=self.add_image
        )
        self.add_btn.pack(side="left")
        tk.Button(
            btn_frame, text="Next", bg="#FECACA", command=self.load_next_or_finish
        ).pack(side="left")
        tk.Button(
            btn_frame, text="Resample", bg="#BFDBFE", command=self.make_rnd_image
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
                self.dc_selection = True
        assert self.label is not None
        assert self.selection is not None
        title = self.title.format(
            cam=self.selection.cam,
            kp=self.selection.kp.cfg.label,
            selection=self.label,
        )
        self.window.title(title)
        self.update_add_button_visibility()
        self.make_rnd_image()
