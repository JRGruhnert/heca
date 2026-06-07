from dataclasses import dataclass
from PIL import Image, ImageTk
import tkinter as tk

from heca.entities.entity import Entity
from heca.image_encoders.image_encoder import ImageEncoder
from heca.misc.base import Configurable
from heca.environment.scenes.scene import Scene


class EncodingViewer(Configurable):
    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        scene: Scene.Config
        kp_encoder: ImageEncoder.Config
        state_encoder: ImageEncoder.Config
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
        self.window.title("Encoding Viewer")

        self.left_canvas = tk.Canvas(
            self.window,
            width=self.cfg.vis_size[0],
            height=self.cfg.vis_size[1],
        )
        self.right_canvas = tk.Canvas(
            self.window,
            width=self.cfg.vis_size[0],
            height=self.cfg.vis_size[1],
        )
        btn_frame = tk.Frame(self.window)
        btn_frame.pack(side="top")
        self.resample_btn = tk.Button(
            btn_frame,
            text="Resample",
            bg="#BFDBFE",
            command=self.on_resample_btn,
        )
        self.resample_btn.pack(side="left")

        self.left_entity_markers: dict[Entity, int] = {}
        self.right_entity_markers: dict[Entity, int] = {}

    def run(self):
        self.on_resample_btn()
        self.window.mainloop()

    def place_marker(self, x: int, y: int, color: str, canvas: tk.Canvas) -> int:
        return canvas.create_oval(
            x - self.cfg.marker_radius,
            y - self.cfg.marker_radius,
            x + self.cfg.marker_radius,
            y + self.cfg.marker_radius,
            fill=color,
        )

    def make_rnd_image(self):
        obs = self.scene.sample_image()
        (s_scene, s_images), (g_scene, g_images) = self.scene.sample_task()

        self.display_img = self.img.resize(
            (self.display_w, self.display_h),
            Image.Resampling.NEAREST,
        )
        self.img_tk = ImageTk.PhotoImage(self.display_img)

        self.canvas.create_image(
            self.offset_x,
            self.offset_y,
            anchor="nw",
            image=self.img_tk,
        )

    def on_resample_btn(self):
        self.canvas.delete("all")
        self.make_rnd_image()

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
