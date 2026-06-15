from dataclasses import dataclass
import numpy as np
import h5py
import matplotlib.pyplot as plt

from matplotlib.widgets import (
    Slider,
    Button,
    TextBox,
)

from heca.agents.experts.expert import ExpertAgent
from heca.misc.base import Configurable


class DemoPostProcessor(Configurable):
    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        agent: ExpertAgent.Config
        file_name: str = "demos.h5"
        save_name: str = "demos_post.h5"

    def __init__(self, cfg: Config):
        self.cfg = cfg

        self.load_path = ExpertAgent.resolve(cfg.agent) / cfg.file_name
        self.save_path = ExpertAgent.resolve(cfg.agent) / cfg.save_name

        self.file = h5py.File(self.load_path, "r")

        self.data = {}
        for k in self.file.keys():
            self.data[k] = np.asarray(self.file[k])

        self.demo_ids = self.data["demo"]

        self.demo_slices = self._build_demo_slices()

        self.demo_idx = 0
        self.frame_idx = 0

    def _build_demo_slices(self):
        slices = []

        ids = self.demo_ids
        unique_ids = np.unique(ids)

        for demo_id in unique_ids:
            idxs = np.where(ids == demo_id)[0]
            slices.append(
                {
                    "id": int(demo_id),
                    "start": idxs[0],
                    "end": idxs[-1],
                }
            )

        return slices

    def _demo_length(self):
        s = self.demo_slices[self.demo_idx]
        return s["end"] - s["start"] + 1

    def _current_global_idx(self):
        s = self.demo_slices[self.demo_idx]
        return s["start"] + self.frame_idx

    def build_ui(self):
        self.fig = plt.figure(figsize=(15, 8))

        self.ax_img = self.fig.add_axes((0.45, 0.05, 0.52, 0.9))
        self.ax_img.axis("off")

        ax_demo = self.fig.add_axes((0.05, 0.90, 0.30, 0.04))
        self.demo_slider = Slider(
            ax_demo,
            "Demo",
            0,
            max(0, len(self.demo_slices) - 1),
            valinit=0,
            valstep=1,
        )

        ax_frame = self.fig.add_axes((0.05, 0.82, 0.30, 0.04))
        self.frame_slider = Slider(
            ax_frame,
            "Frame",
            0,
            max(0, self._demo_length() - 1),
            valinit=0,
            valstep=1,
        )

        self.demo_slider.on_changed(lambda v: self.on_demo_change(int(v)))

        self.frame_slider.on_changed(lambda v: self.on_frame_change(int(v)))

        ax_repeat = self.fig.add_axes((0.05, 0.68, 0.12, 0.05))
        self.repeat_box = TextBox(
            ax_repeat,
            "Copies",
            initial="10",
        )

        ax_insert = self.fig.add_axes((0.20, 0.68, 0.15, 0.06))
        self.insert_btn = Button(
            ax_insert,
            "Insert Copies",
        )
        self.insert_btn.on_clicked(self.insert_frames)

        ax_delete = self.fig.add_axes((0.05, 0.58, 0.30, 0.06))
        self.delete_btn = Button(
            ax_delete,
            "Delete Frame",
            color="salmon",
        )
        self.delete_btn.on_clicked(self.delete_frame)

        ax_save = self.fig.add_axes((0.05, 0.45, 0.30, 0.08))
        self.save_btn = Button(
            ax_save,
            "Save demos_post.h5",
            color="lightgreen",
        )
        self.save_btn.on_clicked(self.save)

        self.status = self.fig.text(
            0.05,
            0.35,
            "",
            fontsize=12,
        )

        self.on_demo_change(0)

    def on_demo_change(self, demo_idx):
        self.demo_idx = demo_idx
        self.frame_idx = 0

        n = self._demo_length()

        self.frame_slider.valmax = max(0, n - 1)
        self.frame_slider.ax.set_xlim(
            0,
            max(1, n - 1),
        )

        self.frame_slider.set_val(0)

    def on_frame_change(self, frame_idx):
        self.frame_idx = frame_idx

        idx = self._current_global_idx()

        frame = self.data["rgb"][idx]

        self.ax_img.clear()
        self.ax_img.imshow(frame)
        self.ax_img.axis("off")

        self.status.set_text(f"Demo {self.demo_idx} | Frame {frame_idx}")

        self.fig.canvas.draw_idle()

    def insert_frames(self, _):
        copies = int(self.repeat_box.text)

        idx = self._current_global_idx()

        for key in self.data:
            value = self.data[key][idx]

            repeated = np.repeat(
                value[None],
                copies,
                axis=0,
            )

            self.data[key] = np.concatenate(
                [
                    self.data[key][: idx + 1],
                    repeated,
                    self.data[key][idx + 1 :],
                ],
                axis=0,
            )

        self.demo_slices = self._build_demo_slices()
        self.on_demo_change(self.demo_idx)

    def delete_frame(self, _):
        idx = self._current_global_idx()

        for key in self.data:
            self.data[key] = np.delete(
                self.data[key],
                idx,
                axis=0,
            )

        self.demo_slices = self._build_demo_slices()
        self.on_demo_change(self.demo_idx)

    def save(self, _):
        with h5py.File(self.save_path, "w") as f:
            for key, value in self.data.items():
                f.create_dataset(
                    key,
                    data=value,
                    compression="gzip",
                )

        print(f"Saved: {self.save_path}")

    def run(self):
        self.build_ui()
        self.on_frame_change(0)
        plt.show()
