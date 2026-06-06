from dataclasses import dataclass
from PIL import Image, ImageTk
import tkinter as tk
from pathlib import Path
from typing import NamedTuple


from heca.entities.entity import Entity
from heca.misc.base import Configurable
from heca.environment.scenes.scene import Scene
from heca.misc import logger


class ImageSelector(Configurable):
    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        scene: Scene.Config
        dc_label: str = "dc_pose"
        cam: str = "front"
        marker_radius: int = 3
        vis_size: tuple[int, int] = (512, 512)
        img_size: tuple[int, int] = (256, 256)

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.scene = Scene.get(self.cfg.scene)
        self.scale = min(
            self.cfg.vis_size[0] / self.cfg.img_size[0],
            self.cfg.vis_size[1] / self.cfg.img_size[1],
        )
        self.display_w = int(self.cfg.img_size[0] * self.scale)
        self.display_h = int(self.cfg.img_size[1] * self.scale)
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
        self.add_btn.pack(side="left")
        self.next_btn.pack(side="left")
        self.resample_btn.pack(side="left")

        self.line_marker: int | None = None
        self.pred_marker: int | None = None
        self.manual_marker: int | None = None
        self.selection: Entity | None = None
        self.sample_idx: int = 0
        self.max_samples_per_label: int = 5
        self.point: tuple[int, int] | None = None
        self.title: str = "Selection for: {entity}.{state} Sample {idx}"
        self.state_samples: dict[Entity, dict[str, list[Image.Image]]] = {
            entity: {state: [] for state in entity.cfg.states}
            for entity in self.scene.entities
        }
        self.kp_samples: dict[Entity, tuple[Image.Image, int, int, int, int]] = {}

        self.selection_iter = iter(self.scene.entities)
        self.label_iter = iter([])
        self.selection = next(self.selection_iter, None)
        self.load_labels_for_selection()

    def run(self):
        self.on_next_btn()
        self.window.mainloop()

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
        elif marker == "all":
            if self.manual_marker is not None:
                self.canvas.delete(self.manual_marker)
            if self.pred_marker is not None:
                self.canvas.delete(self.pred_marker)
            if self.line_marker is not None:
                self.canvas.delete(self.line_marker)
            self.manual_marker = None
            self.pred_marker = None
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

        obs = self.scene.sample_image()
        self.img = Image.fromarray(obs[self.cfg.cam])
        assert (
            self.img.size == self.cfg.img_size
        ), f"Expected image size {self.cfg.img_size}, got {self.img.size}"

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

    def place_relative_marker(self, x: int, y: int, rel_x: int, rel_y: int):
        abs_x = x + rel_x
        abs_y = y + rel_y
        self.place_marker(abs_x, abs_y, "red", "manual")

    def on_canvas_click(self, event: tk.Event):
        assert self.selection is not None
        assert self.label is not None
        if self.label == self.cfg.dc_label:  # else ignore
            self.point = self.scale_point_to_image(event.x, event.y)
            self.place_marker(event.x, event.y, "red", "manual")
            if self.first_click:
                self.first_click = False
                logger.debug(f"First Click point={self.point}")
                x, y = self.point
                self.kp_samples[self.selection] = (self.img, x, y, 0, 0)
                return
            logger.debug(f"Updated point={self.point} for selection {self.selection}")
            img, dc_x, dc_y, _, _ = self.kp_samples[self.selection]
            self.kp_samples[self.selection] = (
                img,
                dc_x,
                dc_y,
                dc_x - self.point[0],
                dc_y - self.point[1],
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

    def on_add_btn(self):
        assert self.selection is not None
        assert self.label is not None

        if self.label == self.cfg.dc_label:
            logger.warning("Cannot add sample for DC label. Ignoring.")
            return

        self.state_samples[self.selection][self.label].append(self.img)
        self.make_rnd_image()

    def load_labels_for_selection(self):
        assert self.selection is not None
        self.label_iter = iter(self.selection.kp.cfg.states)
        self.label = self.cfg.dc_label

    def on_resample_btn(self):
        assert self.selection is not None
        assert self.label is not None
        if self.label == self.cfg.dc_label:
            self.first_click = True
            self.delete_from_canvas("all")

        self.make_rnd_image()

    def on_next_btn(self):
        if self.sample_idx >= self.max_samples_per_label:
            self.sample_idx = 0
            self.label = next(self.label_iter, None)
            if self.label is None:
                self.selection = next(self.selection_iter, None)
                if self.selection is None:
                    self.window.quit()
                    return
                else:
                    self.load_labels_for_selection()
        assert self.label is not None
        assert self.selection is not None
        title = self.title.format(
            cam=self.selection.cam,
            kp=self.selection.kp.cfg.label,
            selection=self.label,
            idx=self.sample_idx,
        )
        self.first_click = True
        self.window.title(title)
        self.make_rnd_image()
