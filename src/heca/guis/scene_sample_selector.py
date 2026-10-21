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
from heca.image_encoders.dino_encoder import DinoEncoder
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


def joint_hint(entity: Entity, name: str) -> str:
    kind = type(entity).__name__.removesuffix("Entity").lower()
    hints = {
        "min": f"click the keypoint with the {kind} joint at its minimum",
        "max": f"click the keypoint with the {kind} joint at its maximum",
        "mid": f"click the keypoint with the {kind} joint at its midpoint",
    }
    return hints.get(name, f"click the keypoint for reference {name!r}")


def save_references(
    scene: Scene, references: dict[str, dict[str, Reference]]
) -> tuple[int, list[str]]:

    merged = {label: dict(group) for label, group in scene.references.items()}
    for label, group in references.items():
        merged.setdefault(label, {}).update(group)
    scene.references = merged
    scene.save()
    scene.load()  # what the next run will see, read the way it will read it

    given = [(label, name) for label, group in references.items() for name in group]
    missing = [
        f"{label}.{name}"
        for label, name in given
        if name not in scene.references.get(label, {})
    ]
    return len(given) - len(missing), missing


def reference_requests(entities: dict[str, Entity]) -> list[Request]:
    requests: list[Request] = []
    for label, entity in entities.items():
        requests.append(
            Request(label, Reference.POSITION, "click the entity's keypoint")
        )
        for name in entity.reference_names:
            requests.append(Request(label, name, joint_hint(entity, name)))
    return requests


class SceneRefSelector(Configurable):
    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        scene: Scene.Config
        dataset_name: str = ""  # empty: the scene's own h5 in its folder
        marker_radius: int = 3
        vis_size: tuple[int, int] = (512, 512)
        kp_extraction: DinoEncoder.Config = DinoEncoder.Config()

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
        self.image_row = tk.Frame(self.window)
        self.image_row.pack(side="top")
        self.map_row = tk.Frame(self.window)
        self.map_row.pack(side="top")
        self.canvas = tk.Canvas(
            self.image_row,
            width=self.cfg.vis_size[0],
            height=self.cfg.vis_size[1],
        )
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.pack(side="left")
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

        self.encoder = DinoEncoder(self.cfg.kp_extraction)
        self.preview_w = self.cfg.vis_size[0]
        self.preview_h = self.cfg.vis_size[1]
        self.preview_canvases: list[tk.Canvas] = []
        self.preview_items: list[int] = []
        self.preview_rings: list[int] = []
        self.preview_photos: list[ImageTk.PhotoImage] = [None, None]  # type: ignore
        for _ in range(2):
            canvas = tk.Canvas(
                self.image_row,
                width=self.preview_w,
                height=self.preview_h,
                bg="black",
            )
            canvas.pack(side="left", padx=2)
            self.preview_canvases.append(canvas)
            self.preview_items.append(canvas.create_image(0, 0, anchor="nw"))
            self.preview_rings.append(
                canvas.create_oval(0, 0, 0, 0, outline="red", width=3)
            )
        self.map_canvases: list[tk.Canvas] = []
        self.map_items: list[int] = []
        self.map_rings: list[int] = []
        self.map_photos: list[ImageTk.PhotoImage] = [None, None, None]  # type: ignore
        for _ in range(3):
            canvas = tk.Canvas(
                self.map_row,
                width=self.preview_w,
                height=self.preview_h,
                bg="black",
            )
            canvas.pack(side="left", padx=2)
            self.map_canvases.append(canvas)
            self.map_items.append(canvas.create_image(0, 0, anchor="nw"))
            self.map_rings.append(
                canvas.create_oval(0, 0, 0, 0, outline="red", width=3)
            )
        self.ref_field = None
        self.heatmap = None
        self.preview_fields: list[np.ndarray] = []
        self.hover: tuple[int, int] | None = None
        self.canvas.bind("<Motion>", self.on_motion)

        self.references: dict[str, dict[str, Reference]] = {
            label: {} for label in self.scene.entities
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
        self.prompt = ""
        self.prompt_item: int | None = None
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        logger.info(
            f"{len(self.requests)} reference(s) to click: "
            f"{', '.join(f'{r.label}.{r.name}' for r in self.requests)}"
        )
        self.on_next_btn()  # show the first request instead of an empty window

    def draw_prompt(self):
        """Keep what to click next on the picture, not just in the title bar."""
        if self.prompt_item is not None:
            self.canvas.delete(self.prompt_item)
        self.prompt_item = self.canvas.create_text(
            8,
            8,
            anchor="nw",
            text=self.prompt,
            fill="#FDE047",
            font=("TkDefaultFont", 13, "bold"),
        )

    def update_title(self, extra: str = ""):
        if self.request is None:
            return
        what = (
            "the entity's position reference"
            if self.request.name == Reference.POSITION
            else f"the joint '{self.request.name}' reference"
        )
        text = self.title.format(
            entity=self.request.label, what=what, hint=self.request.hint
        ) + (f" ({extra})" if extra else "")
        self.window.title(text)
        self.prompt = text
        self.draw_prompt()

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
        self.prompt_item = None
        self.canvas.create_image(
            self.offset_x, self.offset_y, anchor="nw", image=self.img_tk
        )
        self.canvas.create_text(
            8,
            self.display_h - 8,
            anchor="sw",
            text=f"frame {self.frame}",
            fill="#FDE047",
            font=("TkDefaultFont", 10, "bold"),
        )
        self.draw_prompt()
        self.init_previews()

    def unit_field(self, image: Image.Image) -> np.ndarray:
        """The frame's descriptors as (768, H*W) unit columns, for one dot product."""
        field = self.encoder.compute_descriptor(image)[0].numpy()
        field = field.reshape(field.shape[0], -1)
        return field / np.maximum(np.linalg.norm(field, axis=0, keepdims=True), 1e-8)

    def object_travel(self, label: str) -> tuple[np.ndarray, np.ndarray]:
        """Candidate preview frames, and the object's state there relative to the click.

        The state columns are chunked one row at a time, so scattered rows are slow
        and contiguous blocks are cheap: a few windows spread over the dataset hold
        plenty of different object states.
        """
        length = len(self.observations)  # type: ignore
        blocks = [
            (start, min(start + 2000, length))
            for start in [length * i // 5 for i in range(5)]
        ]
        frames = np.concatenate([np.arange(begin, end) for begin, end in blocks])
        columns, anchor = [], []
        for key in self.dataset.keys():
            if not key.startswith(f"heca_{label}_"):
                continue
            if not np.issubdtype(self.dataset[key].dtype, np.number):
                continue
            columns.append(
                np.concatenate(
                    [
                        self.dataset[key][begin:end].reshape(end - begin, -1)
                        for begin, end in blocks
                    ],
                    axis=0,
                ).astype(float)
            )
            anchor.append(np.ravel(self.dataset[key][self.frame]).astype(float))
        values = np.concatenate(columns, axis=1)
        spread = values.std(axis=0)
        varied = spread > 1e-9
        scaled = (values[:, varied] - np.concatenate(anchor)[varied]) / spread[varied]
        return frames, scaled

    def init_previews(self):
        """Descriptors of this frame and of two frames where the object has moved."""
        self.ref_field = self.unit_field(self.img)
        label = self.request.label if self.request else ""
        frames, states = self.object_travel(label)
        distance = np.abs(states).sum(axis=1)
        chosen: list[int] = []
        for _ in range(2):
            live = np.flatnonzero(distance > -1)
            if not len(live):
                break
            top = live[np.argsort(-distance[live])[:200]]
            index = int(random.choice(top))
            chosen.append(int(frames[index]))
            distance[np.abs(frames - frames[index]) < 20] = -1
            distance[np.abs(frames - self.frame) < 20] = -1
            distance = np.minimum(distance, np.abs(states - states[index]).sum(axis=1))
        self.preview_fields = []
        for position, frame in enumerate(chosen):
            image = Image.fromarray(self.observations[frame])  # type: ignore
            self.preview_fields.append(self.unit_field(image))
            photo = ImageTk.PhotoImage(
                image.resize((self.preview_w, self.preview_h), Image.Resampling.NEAREST)
            )
            self.preview_photos[position] = photo
            canvas = self.preview_canvases[position]
            canvas.itemconfig(self.preview_items[position], image=photo)
            canvas.delete("label")
            canvas.create_text(
                6,
                6,
                anchor="nw",
                text=f"frame {frame}",
                fill="#FDE047",
                font=("TkDefaultFont", 10, "bold"),
                tags="label",
            )
        maps = zip(self.map_canvases, self.map_items, self.map_rings)
        for canvas, item, ring in maps:
            canvas.itemconfig(item, image="")
            canvas.coords(ring, 0, 0, 0, 0)

    def on_motion(self, event: tk.Event):
        if self.ref_field is None:
            return
        x, y = self.scale_point_to_image(event.x, event.y)
        if self.hover and abs(x - self.hover[0]) < 2 and abs(y - self.hover[1]) < 2:
            return
        self.hover = (x, y)
        width, height = self.scene.cfg.width, self.scene.cfg.height
        x = min(max(x, 0), width - 1)
        y = min(max(y, 0), height - 1)
        query = self.ref_field[:, y * width + x]

        if self.heatmap is None:
            from matplotlib.colors import LinearSegmentedColormap

            self.heatmap = LinearSegmentedColormap.from_list(
                "heca_heat",
                ["#08131f", "#1b4f79", "#2a8f8a", "#5fae52", "#ef8b2c"],
            )

        for index, field in enumerate([self.ref_field, *self.preview_fields]):
            similarity = (field.T @ query).reshape(height, width)
            normed = (similarity - similarity.min()) / (
                similarity.max() - similarity.min() + 1e-8
            )
            rgb = (self.heatmap(normed)[..., :3] * 255).astype(np.uint8)
            photo = ImageTk.PhotoImage(
                Image.fromarray(rgb).resize(
                    (self.preview_w, self.preview_h), Image.Resampling.BILINEAR
                )
            )
            self.map_photos[index] = photo
            canvas = self.map_canvases[index]
            canvas.itemconfig(self.map_items[index], image=photo)
            peak_y, peak_x = np.unravel_index(
                int(similarity.argmax()), similarity.shape
            )
            px = peak_x * self.preview_w / width
            py = peak_y * self.preview_h / height
            radius = 11
            box = (px - radius, py - radius, px + radius, py + radius)
            canvas.coords(self.map_rings[index], *box)
            if index:
                self.preview_canvases[index - 1].coords(
                    self.preview_rings[index - 1], *box
                )

    def scale_point_to_image(self, x: int, y: int) -> tuple[int, int]:
        return (
            int((x - self.offset_x) / self.scale),
            int((y - self.offset_y) / self.scale),
        )

    def on_canvas_click(self, event: tk.Event):
        if self.request is None:
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
        self.pending = Reference(
            image=self.img, x=x, y=y, xyz=xyz, frame=int(self.frame)
        )
        self.hover = None
        self.on_motion(event)
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
        assert self.pending is not None, "Click the keypoint in the image first"
        self.references[request.label][request.name] = self.pending
        logger.info(
            f"{request.label}.{request.name}: pixel ({self.pending.x}, "
            f"{self.pending.y}) at {np.round(self.pending.xyz, 4)}"
        )
        self.on_next_btn()

    def on_next_btn(self):
        if (
            self.request is not None
            and self.request.name not in self.references[self.request.label]
        ):
            self.update_title("nothing recorded yet: click the keypoint, then Add")
            return
        self.request = next(self.selection_iter, None)
        if self.request is None:
            self.finish()
            return
        self.pending = None
        self.update_title()
        self.on_resample_btn()

    def run(self):
        self.window.mainloop()

    def on_close(self):
        """Closing the window keeps whatever was clicked rather than dropping it."""
        clicked = {
            label: dict(group) for label, group in self.references.items() if group
        }
        if clicked:
            recorded, missing = save_references(self.scene, clicked)
            logger.warning(
                f"window closed before the end: saved the {recorded} reference(s) "
                f"that were clicked"
                + (f", {len(missing)} did not read back" if missing else "")
            )
        else:
            logger.warning(
                "window closed with nothing recorded: the scene is unchanged"
            )
        self.window.destroy()

    def finish(self):
        recorded, missing = save_references(self.scene, self.references)
        if missing:
            logger.error(
                f"{len(missing)} reference(s) did not read back after saving: "
                f"{', '.join(missing)}"
            )
        if not self.scene.references_ready():
            logger.error(
                "the reference set is still incomplete; run this again before "
                "using the scene"
            )
        else:
            logger.info(
                f"saved {recorded} reference(s) to {Scene.save_dir(self.scene.cfg)}"
            )
        self.window.quit()
        self.window.destroy()
        sys.exit(0)
