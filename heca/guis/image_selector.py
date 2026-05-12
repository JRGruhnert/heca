from dataclasses import dataclass
import os
from PIL import Image, ImageTk
import tkinter as tk


from heca.classes.config import Configurable
from heca.environment.scenes.scene import Scene


class ImageSelector(Configurable):
    @dataclass(kw_only=True)
    class Config(Configurable.Config):
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
        self.canvas.pack()
        button_frame = tk.Frame(self.window)
        button_frame.pack(side="top")

        tk.Button(button_frame, text="Add", command=self.add).pack(side="left")
        tk.Button(button_frame, text="Next", command=self.next).pack(side="left")
        tk.Button(button_frame, text="Resample", command=self.refresh).pack(side="left")

        # Prepare selection tuples
        self.selection_labels = ["pose", "state"]
        # self.cam_labels = self.get_cam_labels()
        self.cam_labels = ["front"]
        self.selection_tuples: list[tuple[str, str, str]] = []
        for entity in self.scene.entities:
            print(entity.cfg.label)
            print(entity.cfg.states)
            for state in entity.cfg.states:
                for cam in self.cam_labels:
                    self.selection_tuples.append((cam, entity.cfg.label, state))

        self.selection_iter = iter(self.selection_tuples)

        self.current_tuple = None
        self.title = "Selection for: {cam}.{entity}.{selection}"
        self.current_sample = 0
        self.load_next_or_finish()

    def load_image(self):
        assert self.current_tuple is not None
        nd_pictures = self.scene.sample_images()
        picture = nd_pictures[self.current_tuple[0]]
        self.img = Image.fromarray(picture)
        self.img_tk = ImageTk.PhotoImage(self.img)
        self.canvas.delete("all")
        self.canvas.create_image(
            0,
            0,
            anchor="nw",
            image=self.img_tk,
        )

    def add(self):
        assert self.current_tuple is not None
        cam, entity_label, s_label = self.current_tuple
        save_dir = f"data/samples/{cam}"
        os.makedirs(save_dir, exist_ok=True)
        self.img.save(f"{save_dir}/{entity_label}_{s_label}_{self.current_sample}.png")
        self.current_sample += 1
        self.refresh()

    def next(self):
        self.load_next_or_finish()

    def save(self):
        pass

    def run(self):
        self.window.mainloop()

    def get_cam_labels(self) -> list[str]:
        cam_recordings = self.scene.sample_td_images()
        return list(cam_recordings.records.keys())

    def load_next_or_finish(self):
        self.current_tuple = next(self.selection_iter, None)
        self.current_sample = 0
        if self.current_tuple is not None:
            title = self.title.format(
                cam=self.current_tuple[0],
                entity=self.current_tuple[1],
                selection=self.current_tuple[2],
            )
            self.window.title(title)
            self.load_image()
        else:
            self.save()
            self.window.quit()

    def refresh(self):
        self.load_image()


from heca.environment.scenes.calvin.scene import CalvinScene

selector_cfg = ImageSelector.Config(
    scene=CalvinScene.Query(),
)
selector = ImageSelector.create(selector_cfg)
selector.run()
