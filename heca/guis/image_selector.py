from dataclasses import dataclass
from PIL import Image, ImageTk
import tkinter as tk
import numpy as np
import torch

from heca.classes.config import Configurable
from heca.environment.scenes.scene import Scene
from heca.misc.td import TDSceneReferences


class ImageSelector(Configurable):
    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        marker_radius: int = 5
        scene: Scene.Query

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.scene = Scene.search(self.cfg.scene)
        self.window = tk.Tk()
        self.canvas = tk.Canvas(
            self.window,
            width=self.scene.cfg.img_size[0],
            height=self.scene.cfg.img_size[1],
        )
        # controls
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.pack()
        tk.Button(self.window, text="Accept", command=self.accept).pack()
        tk.Button(self.window, text="Resample", command=self.refresh).pack()

        # Prepare selection tuples
        self.selection_labels = ["pose", "state"]
        self.cam_labels = self.get_cam_labels()
        self.selection_tuples: list[tuple[str, str, str]] = []
        for entity in self.scene.entities:
            for s_label in self.selection_labels:
                for cam in self.cam_labels:
                    self.selection_tuples.append((cam, entity.cfg.label, s_label))

        self.selection_iter = iter(self.selection_tuples)
        self.result: TDSceneReferences = TDSceneReferences()

        self.point = None
        self.marker = None
        self.current_tuple = None
        self.image = None

        self.load_next_or_finish()

    def torch_to_imagetk(self, img_tensor: torch.Tensor) -> ImageTk.PhotoImage:
        # Ensure tensor is on CPU and detached
        img_tensor = img_tensor.cpu().detach()
        # If float, scale to 0-255 and convert to uint8
        if img_tensor.dtype != torch.uint8:
            img_tensor = (img_tensor * 255).clamp(0, 255).to(torch.uint8)
        # Convert to numpy and (H, W, C)
        img_np = img_tensor.numpy()
        if img_np.shape[0] == 3:  # (C, H, W) -> (H, W, C)
            img_np = np.transpose(img_np, (1, 2, 0))
        img_pil = Image.fromarray(img_np)
        return ImageTk.PhotoImage(img_pil)

    def load_image(self):
        assert self.current_tuple is not None
        recordings = self.scene.sample_images()
        # TODO: get image from here
        # x is a dict[str, np.ndarray] which has to be converted to tensor
        self.img_raw = recordings.records[self.current_tuple[0]].rgb
        self.img = self.torch_to_imagetk(self.img_raw)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.img)

        self.point = None
        self.marker = None

    def accept(self):
        assert self.point is not None
        assert self.current_tuple is not None
        cam, entity_label, s_label = self.current_tuple

        if s_label == "pose":
            self.result.add_pose(
                cam_label=cam,
                entity_label=entity_label,
                point=torch.tensor(self.point, dtype=torch.float32),
                img_raw=self.img_raw,
            )
        elif s_label == "state":
            self.result.add_state(
                cam_label=cam,
                entity_label=entity_label,
                img_raw=self.img_raw,
            )
        else:
            raise ValueError(f"Unknown selection label: {s_label}")

        self.load_next_or_finish()

    def save(self):
        pass

    def run(self):
        self.window.mainloop()

    def get_cam_labels(self) -> list[str]:
        cam_recordings = self.scene.sample_images()
        return list(cam_recordings.records.keys())

    def load_next_or_finish(self):
        self.current_tuple = next(self.selection_iter, None)
        if self.current_tuple is not None:
            title = f"Selection for: {self.current_tuple[0]}.{self.current_tuple[1]}.{self.current_tuple[2]}"
            self.window.title(title)
            self.load_image()
        else:
            self.save()
            self.window.quit()

    def refresh(self):

        self.load_image()

    def on_click(self, event: tk.Event):
        self.point = (event.x, event.y)
        self.place_marker(self.point)

    def place_marker(self, point: tuple[int, int]):
        if self.marker:
            self.canvas.delete(self.marker)

        self.marker = self.canvas.create_oval(
            point[0] - self.cfg.marker_radius,
            point[1] - self.cfg.marker_radius,
            point[0] + self.cfg.marker_radius,
            point[1] + self.cfg.marker_radius,
            fill="red",
        )


from heca.environment.scenes.calvin.scene import CalvinScene

selector_cfg = ImageSelector.Config(
    scene=CalvinScene.Query(),
)
selector = ImageSelector.create(selector_cfg)
selector.run()
