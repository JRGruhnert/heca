from dataclasses import dataclass
from PIL import Image, ImageTk
import tkinter as tk
import numpy as np
import torch

from heca.classes.config import Configurable
from heca.entities.entity import Entity
from heca.environment.scenes.calvin.scene import CalvinScene
from heca.environment.scenes.scene import Scene
from tapas_gmm_modified.dense_correspondence.correspondence_finder import (
    find_best_match,
)
from tapas_gmm_modified.dense_correspondence.correspondence_augmentation import (
    random_flip,
)


class ImageSelector(Configurable):
    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        marker_radius: int = 5
        samples_per_point: int = 3
        scene: Scene.Query

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.root = tk.Tk()
        self.root.title("Image Selector")
        self.canvas = tk.Canvas(self.root, width=400, height=400)
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.pack()
        self.scene = Scene.search(self.cfg.scene)

        self.selection_pairs: list[tuple[Entity.Config, str]] = []
        for query in self.scene.entities():
            entity = Entity.search(query)
            for key in entity.properties.keys():
                self.selection_pairs.append((query, key))

        self.pair_iter = iter(self.selection_pairs)
        self.results: dict[
            Entity.Query,
            dict[str, list[tuple[tuple[int, int], np.ndarray]]],
        ] = {}

        self.current_pair = None
        self.point = None
        self.marker = None
        self.sample = 0

        # controls
        tk.Button(self.root, text="Accept", command=self.accept).pack()
        tk.Button(self.root, text="Resample", command=self.refresh).pack()

        self.next_pair()  # Start with the first (entity, property)

    def set_canvas(self):
        self.canvas = tk.Canvas(self.root, width=400, height=400)
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.pack()

    def load_image(self):
        x = self.scene.sample_images()
        # TODO: get image from here
        # x is a dict[str, np.ndarray] which has to be converted to tensor
        self.img_raw = np.random.randint(0, 255, (3, 400, 400), dtype=np.uint8)

        img = Image.fromarray(self.img_raw.transpose(1, 2, 0))
        self.img = ImageTk.PhotoImage(img)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.img)

        self.point = None
        self.marker = None

        if self.current_pair is not None:
            title = f"Image Selector - Entity: {self.current_pair[0].label}, Property: {self.current_pair[1]}"
            self.root.title(title)

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

    def next_pair(self):
        try:
            self.current_pair = next(self.pair_iter)
            self.sample = 0
            self.load_image()
        except StopIteration:
            self.current_pair = None
            self.save()
            self.root.quit()

    def accept(self):
        assert self.current_pair is not None
        assert self.point is not None
        if self.current_pair is not None:
            entity_query, prop_name = self.current_pair
            if entity_query not in self.results:
                self.results[entity_query] = {}
            if prop_name not in self.results[entity_query]:
                self.results[entity_query][prop_name] = []

            self.results[entity_query][prop_name].append((self.point, self.img_raw))
            self.sample += 1
            if self.sample < self.cfg.samples_per_point:
                self.load_image()
                return
        self.next_pair()

    def refresh(self):
        self.load_image()

    def save(self):
        for entity, prop_points in self.results.items():
            for prop, point in prop_points.items():
                print(f"Entity: {entity}, Property: {prop}, Selected Point: {point}")

        # Dummy descriptors for demonstration (replace with real data)
        dummy_res_a = np.random.rand(400, 400, 16)
        dummy_res_b = np.random.rand(400, 400, 16)
        dummy_img = np.random.randint(0, 255, (3, 400, 400), dtype=np.uint8)
        metric = "euclidean"
        for entity, prop_points in self.results.items():
            for prop, point in prop_points.items():
                if point is not None:
                    img_tensor = torch.from_numpy(dummy_img)
                    u, v = point
                    uv_pixel_positions = (torch.tensor([u]), torch.tensor([v]))
                    aug_images, aug_uv = random_flip([img_tensor], uv_pixel_positions)
                    aug_point = (int(aug_uv[0][0]), int(aug_uv[1][0]))

                    # Use augmented point in find_best_match
                    best_match_uv, best_match_diff, norm_diffs = find_best_match(
                        pixel_a=aug_point,
                        res_a=dummy_res_a,
                        res_b=dummy_res_b,
                        metric=metric,
                    )

    def run(self):
        self.root.mainloop()


selector_cfg = ImageSelector.Config(
    scene=CalvinScene.Query(),
)
selector = ImageSelector(cfg=selector_cfg)
selector.run()
