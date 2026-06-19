from dataclasses import dataclass
import random
import sys
from PIL import Image, ImageTk
import tkinter as tk

import h5py
import numpy as np

from heca.entities.entity import Entity
from heca.misc.base import Configurable
from heca.environment.scenes.scene import Scene
from heca.misc import logger


class SceneSampleSelector(Configurable):
    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        scene: Scene.Config
        kp_label: str = "keypoint"
        dataset_name: str = "visual-scene-play-v0.h5"
        marker_radius: int = 3
        sample_count: int = 5
        vis_size: tuple[int, int] = (512, 512)

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.scene = Scene.get(self.cfg.scene, load=False)
        self.scale = min(
            self.cfg.vis_size[0] / self.scene.cfg.width,
            self.cfg.vis_size[1] / self.scene.cfg.height,
        )
        self.display_w = int(self.scene.cfg.width * self.scale)
        self.display_h = int(self.scene.cfg.height * self.scale)
        self.offset_x = (self.cfg.vis_size[0] - self.display_w) // 2
        self.offset_y = (self.cfg.vis_size[1] - self.display_h) // 2

        self.window = tk.Tk()
        self.canvas = tk.Canvas(
            self.window,
            width=self.cfg.vis_size[0],
            height=self.cfg.vis_size[1],
        )
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.pack()
        btn_frame = tk.Frame(self.window)
        btn_frame.pack(side="top")
        self.add_btn = tk.Button(
            btn_frame,
            text="Add",
            bg="#D1FAE5",
            command=self.on_add_btn,
        )
        self.next_btn = tk.Button(
            btn_frame,
            text="Next",
            bg="#FECACA",
            command=self.on_next_btn,
        )
        self.resample_btn = tk.Button(
            btn_frame,
            text="Resample",
            bg="#BFDBFE",
            command=self.on_resample_btn,
        )
        self.clear_vanvas_btn = tk.Button(
            btn_frame,
            text="Clear Canvas",
            bg="#E5E7EB",
            command=self.on_clear_btn,
        )

        self.add_btn.pack(side="left")
        self.next_btn.pack(side="left")
        self.resample_btn.pack(side="left")
        self.clear_vanvas_btn.pack(side="left")

        self.selection: tuple[Entity | None, str] = (None, "")
        self.sample_idx: int = 0

        self.line_marker: int | None = None
        self.kp_marker: int | None = None
        self.state_marker: int | None = None
        self.title: str = "Selection for: {entity}.{state} Sample {idx}"
        self.entities = [self.scene.cursor] + self.scene.entities
        self.state_samples: dict[str, dict[str, list[Image.Image]]] = {
            entity.cfg.label: {state: [] for state in entity.cfg.states}
            for entity in self.entities
        }
        self.kp_samples: dict[str, tuple[Image.Image, int, int, int, int]] = {}
        selection_tuples: list[tuple[Entity, str]] = []
        for entity in self.entities:
            selection_tuples.append((entity, self.cfg.kp_label))
            for state in entity.cfg.states:
                selection_tuples.append((entity, state))
        self.selection_iter = iter(selection_tuples)

        self.load_path = Scene.resolve_base(cfg.scene) / "demos" / cfg.dataset_name
        self.train_dataset = h5py.File(self.load_path, "r")
        self.observations = self.train_dataset["rgb"]  # type: ignore

    def run(self):
        self.on_next_btn()
        self.window.mainloop()

    def delete_from_canvas(self, marker: str):
        if marker == "manual":
            if self.state_marker is not None:
                self.canvas.delete(self.state_marker)
            self.state_marker = None
        elif marker == "pred":
            if self.kp_marker is not None:
                self.canvas.delete(self.kp_marker)
            self.kp_marker = None
        elif marker == "line":
            if self.line_marker is not None:
                self.canvas.delete(self.line_marker)
            self.line_marker = None
        elif marker == "all":
            if self.state_marker is not None:
                self.canvas.delete(self.state_marker)
            if self.kp_marker is not None:
                self.canvas.delete(self.kp_marker)
            if self.line_marker is not None:
                self.canvas.delete(self.line_marker)
            self.state_marker = None
            self.kp_marker = None
            self.line_marker = None
        else:
            logger.warning(f"Unknown marker type: {marker}")

    def place_line_marker(self):
        if self.state_marker is None or self.kp_marker is None:
            return

        mx1, my1, mx2, my2 = self.canvas.coords(self.state_marker)
        px1, py1, px2, py2 = self.canvas.coords(self.kp_marker)

        mx = (mx1 + mx2) / 2.0
        my = (my1 + my2) / 2.0
        px = (px1 + px2) / 2.0
        py = (py1 + py2) / 2.0

        self.line_marker = self.canvas.create_line(
            mx, my, px, py, fill="yellow", width=2
        )

    def place_marker(self, x: int, y: int, color: str) -> int:
        return self.canvas.create_oval(
            x - self.cfg.marker_radius,
            y - self.cfg.marker_radius,
            x + self.cfg.marker_radius,
            y + self.cfg.marker_radius,
            fill=color,
        )

    def make_rnd_image(self):
        self.kp_marker = None
        self.state_marker = None
        self.line_marker = None
        idx = random.randint(0, len(self.observations) - 1)  # type: ignore
        obs: np.ndarray = self.observations[idx]  # type: ignore
        self.img = Image.fromarray(obs)

        self.display_img = self.img.resize(
            (self.display_w, self.display_h),
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

    def on_canvas_click(self, event: tk.Event):
        assert self.selection[0] is not None
        point = self.scale_point_to_image(event.x, event.y)

        if self.kp_marker is not None:
            # Kp Marker already placed
            img, dc_x, dc_y, _, _ = self.kp_samples[self.selection[0].cfg.label]
            self.kp_samples[self.selection[0].cfg.label] = (
                img,
                dc_x,
                dc_y,
                dc_x - point[0],
                dc_y - point[1],
            )
            self.state_marker = self.place_marker(event.x, event.y, "red")
            self.place_line_marker()
        else:
            # Kp Marker not placed yet, place it and save the image and point
            self.kp_marker = self.place_marker(event.x, event.y, "blue")
            self.kp_samples[self.selection[0].cfg.label] = (
                self.img,
                point[0],
                point[1],
                0,
                0,
            )

    def on_clear_btn(self):
        self.delete_from_canvas("all")
        if self.selection[0] is not None and self.selection[1] == self.cfg.kp_label:
            self.kp_samples.pop(self.selection[0].cfg.label)

    def on_resample_btn(self):
        self.delete_from_canvas("all")
        self.make_rnd_image()

    def on_add_btn(self):
        entity = self.selection[0]
        state = self.selection[1]
        assert entity is not None, "No entity selected"
        assert state != "", "No state selected"
        assert state != self.cfg.kp_label, "Should not happen"
        elabel = entity.cfg.label
        self.state_samples[elabel][state].append(self.img)
        idx = len(self.state_samples[elabel][state])
        if idx >= self.cfg.sample_count:
            self.on_next_btn()
        else:
            self.update_title(entity=elabel, state=state, idx=idx)
            self.on_resample_btn()

    def on_next_btn(self):
        self.load_next_selection()
        assert self.selection[0] is not None, "Should not happen"
        if self.selection[1] == self.cfg.kp_label:
            self.add_btn.config(state="disabled")
        else:
            self.add_btn.config(state="normal")
        self.update_title(
            entity=self.selection[0].cfg.label,
            state=self.selection[1],
            idx=0,
        )
        self.on_resample_btn()

    def update_title(self, entity: str, state: str, idx: int):
        title = self.title.format(
            entity=entity,
            state=state,
            idx=idx,
        )
        self.window.title(title)

    def load_next_selection(self):
        self.selection = next(self.selection_iter, (None, ""))
        if self.selection[0] is None:
            self.scene.kp_references = self.kp_samples
            self.scene.state_references = self.state_samples
            Scene.save(self.scene.cfg)
            self.window.quit()
            self.window.destroy()
            sys.exit(0)
