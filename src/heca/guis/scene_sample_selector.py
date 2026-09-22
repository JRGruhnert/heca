from dataclasses import dataclass
import random
import sys
from pathlib import Path
from PIL import Image, ImageTk
import tkinter as tk

import h5py
import numpy as np

from heca.data.entity import Entity
from heca.data.reference import Reference
from heca.image_encoders.image_encoder import ImageEncoder
from heca.misc.base import Configurable
from heca.scenes.scene import Scene
from heca.misc import logger


def demo_dataset(scene_cfg: Scene.Config, name: str = "") -> Path:
    base = Scene.save_dir(scene_cfg)
    if name:
        return base / name
    candidates = sorted(base.glob("*.h5"))
    if not candidates:
        raise FileNotFoundError(f"no .h5 demo dataset in {base}")
    return candidates[0]


@dataclass(frozen=True)
class Request:
    label: str
    name: str
    hint: str
    state: int | None = None


def state_name(idx: int) -> str:
    return f"state{idx}"


def joint_hint(entity: Entity, name: str) -> str:
    kind = type(entity).__name__.removesuffix("Entity").lower()
    hints = {
        "min": f"click the keypoint with the {kind} joint at its minimum",
        "max": f"click the keypoint with the {kind} joint at its maximum",
        "mid": f"click the keypoint with the {kind} joint at its midpoint",
    }
    return hints.get(name, f"click the keypoint for reference {name!r}")


def reference_requests(entities: dict[str, Entity]) -> list[Request]:
    requests: list[Request] = []
    for label, entity in entities.items():
        requests.append(
            Request(label, Reference.POSITION, "click the entity's keypoint")
        )
        for name in entity.reference_names:
            requests.append(Request(label, name, joint_hint(entity, name)))
        if entity.cfg.n_states > 1:
            for idx in range(entity.cfg.n_states):
                requests.append(
                    Request(
                        label, state_name(idx), f"a picture of state {idx}", state=idx
                    )
                )
    return requests


class SceneRefSelector(Configurable):
    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        scene: Scene.Config
        dataset_name: str = ""  # empty: the scene's own h5 in its folder
        marker_radius: int = 3
        sample_count: int = 5
        vis_size: tuple[int, int] = (512, 512)

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.scene = Scene.get(self.cfg.scene, auto_load=False)
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
            btn_frame, text="Add", bg="#D1FAE5", command=self.on_add_btn
        )
        self.next_btn = tk.Button(
            btn_frame, text="Next", bg="#FECACA", command=self.on_next_btn
        )
        self.resample_btn = tk.Button(
            btn_frame, text="Resample", bg="#BFDBFE", command=self.on_resample_btn
        )
        self.clear_btn = tk.Button(
            btn_frame, text="Clear Canvas", bg="#E5E7EB", command=self.on_clear_btn
        )
        for button in (self.add_btn, self.next_btn, self.resample_btn, self.clear_btn):
            button.pack(side="left")

        self.references: dict[str, dict[str, Reference]] = {
            label: {} for label in self.scene.entities
        }
        self.state_samples: dict[str, dict[int, list[Image.Image]]] = {
            label: {idx: [] for idx in range(entity.cfg.n_states)}
            for label, entity in self.scene.entities.items()
        }
        self.requests = reference_requests(self.scene.entities)
        self.selection_iter = iter(self.requests)
        self.request: Request | None = None
        self.pending: Reference | None = None
        self.kp_marker: int | None = None
        self.frame: int = 0
        self.img: Image.Image = Image.new(
            "RGB", (self.scene.cfg.width, self.scene.cfg.height)
        )

        self.load_path = demo_dataset(self.cfg.scene, self.cfg.dataset_name)
        self.dataset = h5py.File(self.load_path, "r")
        self.observations = self.dataset["rgb"]
        self.depth = self.dataset["depth"]
        self.extrinsics = self.dataset["extrinsics"]
        self.intrinsics = self.dataset["intrinsics"]
        self.title = "{entity}: add {what} - {hint}"

    def update_title(self, extra: str = ""):
        if self.request is None:
            return
        what = (
            "the entity's position reference"
            if self.request.name == Reference.POSITION
            else (
                f"state {self.request.state} sample"
                if self.request.state is not None
                else f"the joint '{self.request.name}' reference"
            )
        )
        self.window.title(
            self.title.format(
                entity=self.request.label, what=what, hint=self.request.hint
            )
            + (f" ({extra})" if extra else "")
        )

    def place_marker(self, x: int, y: int) -> int:
        r = self.cfg.marker_radius
        return self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="blue")

    def make_rnd_image(self):
        self.kp_marker = None
        self.frame = random.randint(0, len(self.observations) - 1)  # type: ignore
        self.img = Image.fromarray(self.observations[self.frame])  # type: ignore
        self.img_tk = ImageTk.PhotoImage(
            self.img.resize((self.display_w, self.display_h), Image.Resampling.NEAREST)
        )
        self.canvas.delete("all")
        self.canvas.create_image(
            self.offset_x, self.offset_y, anchor="nw", image=self.img_tk
        )

    def scale_point_to_image(self, x: int, y: int) -> tuple[int, int]:
        return (
            int((x - self.offset_x) / self.scale),
            int((y - self.offset_y) / self.scale),
        )

    def on_canvas_click(self, event: tk.Event):
        if self.request is None or self.request.state is not None:
            # A state sample is a whole picture: the frame is the sample, so a
            # click on it means nothing.
            return
        if self.kp_marker is not None:
            self.canvas.delete(self.kp_marker)
        self.kp_marker = self.place_marker(event.x, event.y)
        x, y = self.scale_point_to_image(event.x, event.y)
        xyz = ImageEncoder.pixel_to_world(
            y,
            x,
            self.depth[self.frame],  # type: ignore
            self.extrinsics[self.frame],  # type: ignore
            self.intrinsics[self.frame],  # type: ignore
        )
        self.pending = Reference(image=self.img, x=x, y=y, xyz=xyz)
        self.update_title(f"pixel {x}, {y} -> {np.round(xyz, 3)}")

    def on_clear_btn(self):
        if self.kp_marker is not None:
            self.canvas.delete(self.kp_marker)
            self.kp_marker = None
        self.pending = None
        self.update_title()

    def on_resample_btn(self):
        self.on_clear_btn()
        self.make_rnd_image()

    def on_add_btn(self):
        request = self.request
        assert request is not None, "No reference in progress"
        if request.state is not None:
            self.state_samples[request.label][request.state].append(self.img)
            count = len(self.state_samples[request.label][request.state])
            if count >= self.cfg.sample_count:
                self.on_next_btn()
            else:
                self.update_title(f"{count}/{self.cfg.sample_count}")
                self.on_resample_btn()
            return

        assert self.pending is not None, "Click the keypoint in the image first"
        self.references[request.label][request.name] = self.pending
        logger.info(
            f"{request.label}.{request.name}: pixel ({self.pending.x}, "
            f"{self.pending.y}) at {np.round(self.pending.xyz, 4)}"
        )
        self.on_next_btn()

    def on_next_btn(self):
        self.request = next(self.selection_iter, None)
        if self.request is None:
            self.finish()
            return
        self.pending = None
        self.update_title()
        self.on_resample_btn()

    def finish(self):
        self.scene.references = self.references
        self.scene.state_references = self.state_samples
        self.scene.save()
        logger.info(
            f"saved references for {len(self.references)} entities to "
            f"{Scene.save_dir(self.scene.cfg)}"
        )
        self.window.quit()
        self.window.destroy()
        sys.exit(0)
