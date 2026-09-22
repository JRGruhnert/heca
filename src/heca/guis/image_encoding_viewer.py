from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageTk
import numpy as np
import tkinter as tk

from heca.image_encoders.image_encoder import ImageEncoder
from heca.misc.base import Configurable
from heca.misc import logger
from heca.scenes.scene import Scene


class ImageEncodingViewer(Configurable):
    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        scene: Scene.Config
        kp_encoder: ImageEncoder.Config
        state_encoder: ImageEncoder.Config
        marker_radius: int = 4
        vis_size: tuple[int, int] = (512, 512)

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.scene = Scene.get(self.cfg.scene)
        if not self.scene.references_ready():
            raise RuntimeError(
                f"{self.cfg.scene.tag}: no complete reference set; run "
                "scripts/d01_select_references.py first"
            )
        self.kp_encoder = ImageEncoder.get(self.cfg.kp_encoder)
        self.state_encoder = ImageEncoder.get(self.cfg.state_encoder)
        self.kp_encoder.prepare_for_scene(self.cfg.scene)
        self.state_encoder.prepare_for_scene(self.cfg.scene)

        self.scale = min(
            self.cfg.vis_size[0] / self.scene.cfg.width,
            self.cfg.vis_size[1] / self.scene.cfg.height,
        )
        self.display_w = int(self.scene.cfg.width * self.scale)
        self.display_h = int(self.scene.cfg.height * self.scale)
        self.offset_x = (self.cfg.vis_size[0] - self.display_w) // 2
        self.offset_y = (self.cfg.vis_size[1] - self.display_h) // 2

        self.window = tk.Tk()
        self.window.title("Encoding Viewer")
        btn_frame = tk.Frame(self.window)
        btn_frame.pack(side="top")
        tk.Button(
            btn_frame, text="Resample", bg="#BFDBFE", command=self.on_resample_btn
        ).pack(side="left")
        self.canvases = [
            tk.Canvas(
                self.window, width=self.cfg.vis_size[0], height=self.cfg.vis_size[1]
            )
            for _ in range(2)
        ]
        for canvas in self.canvases:
            canvas.pack(side="left")
        self.images: list[ImageTk.PhotoImage] = []

    def run(self):
        self.on_resample_btn()
        self.window.mainloop()

    @staticmethod
    def as_array(image) -> np.ndarray:
        rgb = image.rgb
        if hasattr(rgb, "permute"):
            rgb = rgb.permute(1, 2, 0).numpy()
        return np.asarray(np.asarray(rgb) * 255.0, dtype=np.uint8)

    def show(self, canvas: tk.Canvas, picture: Image.Image, lines: list[str]):
        canvas.delete("all")
        display = picture.resize(
            (self.display_w, self.display_h), Image.Resampling.NEAREST
        )
        self.images.append(ImageTk.PhotoImage(display))
        canvas.create_image(
            self.offset_x, self.offset_y, anchor="nw", image=self.images[-1]
        )
        canvas.create_text(
            8,
            8,
            anchor="nw",
            text="\n".join(lines),
            fill="yellow",
            font=("TkDefaultFont", 8),
        )

    def annotate(self, tdimage, note: str) -> tuple[Image.Image, list[str]]:
        poses = self.kp_encoder.extract_poses(tdimage)
        states = self.state_encoder.extract_states(tdimage)
        picture = Image.fromarray(self.as_array(tdimage))
        draw = ImageDraw.Draw(picture)
        lines = [note]
        for label, keypoint in poses.items():
            x, y = keypoint.x * self.scale, keypoint.y * self.scale
            r = self.cfg.marker_radius
            colour = (
                "lime"
                if keypoint.score >= self.cfg.kp_encoder.kp_selection_threshold
                else "red"
            )
            draw.ellipse((x - r, y - r, x + r, y + r), outline=colour, width=2)
            state = f" state {states[label][0]}" if label in states else ""
            draw.text((x + r + 2, y), f"{label}{state}", fill=colour)
            lines.append(f"{label}: score {keypoint.score:.2f}{state}")
        return picture, lines

    def on_resample_btn(self):
        (_, start_image), (_, goal_image) = self.scene.sample_task()
        logger.info("sampled a task, reading both encoders")
        for canvas, image, note in (
            (self.canvases[0], start_image, "start"),
            (self.canvases[1], goal_image, "goal"),
        ):
            # the two frames are different scenes, not consecutive ones
            self.kp_encoder.reset_tracking()
            picture, lines = self.annotate(image, note)
            self.show(canvas, picture, lines)
