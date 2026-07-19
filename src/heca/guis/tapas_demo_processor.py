from dataclasses import dataclass
import numpy as np
import h5py
import matplotlib.pyplot as plt

from matplotlib.widgets import Slider, Button, TextBox

from heca.agents.experts.expert import ExpertAgent
from heca.misc.base import Configurable


class TapasDemoProcessor(Configurable):
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
        self.demo_slices = self._build_demo_slices()

        self.demo_idx = 0
        self.frame_idx = 0
        self.key_pressed = set()

    def _build_demo_slices(self):
        slices = []

        ids = self.data["demo"]
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
        self.fig = plt.figure(figsize=(14, 8))
        self.fig.canvas.manager.set_window_title(f"Demo Postprocessor: {self.cfg.agent.folder}")  # type: ignore

        # Image panel (right)
        self.ax_img = self.fig.add_axes((0.45, 0.08, 0.52, 0.88))
        self.ax_img.axis("off")

        ax_demo = self.fig.add_axes((0.08, 0.88, 0.30, 0.04))
        self.demo_slider = Slider(
            ax_demo,
            "Demo",
            0,
            max(0, len(self.demo_slices) - 1),
            valinit=0,
            valstep=1,
        )

        ax_frame = self.fig.add_axes((0.08, 0.78, 0.30, 0.04))
        self.frame_slider = Slider(
            ax_frame,
            "Frame",
            0,
            max(0, self._demo_length() - 1),
            valinit=0,
            valstep=1,
        )

        ax_repeat = self.fig.add_axes((0.05, 0.68, 0.12, 0.06))
        self.repeat_box = TextBox(
            ax_repeat,
            "Copies",
            initial="5",
        )

        ax_insert = self.fig.add_axes((0.20, 0.68, 0.15, 0.06))
        self.insert_btn = Button(
            ax_insert,
            "Repeat Frame",
        )
        ax_delete = self.fig.add_axes((0.20, 0.58, 0.15, 0.06))
        self.delete_btn = Button(
            ax_delete,
            "Delete Frame",
            color="salmon",
        )
        self.insert_btn.on_clicked(self.insert_frames)
        self.delete_btn.on_clicked(self.delete_frame)
        self.demo_slider.on_changed(lambda v: self.on_demo_change(int(v)))
        self.frame_slider.on_changed(lambda v: self.on_frame_change(int(v)))

        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.fig.canvas.mpl_connect("key_release_event", self._on_key_release)
        self.fig.canvas.mpl_connect("close_event", self._on_close)

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

        self.fig.canvas.draw_idle()

    def insert_frames(self, _):
        copies = int(self.repeat_box.text)

        idx = self._current_global_idx()

        for key in self.data:
            if key == "actions":
                value = np.zeros(5)
            else:
                value = self.data[key][idx]

            repeated = np.repeat(
                value[None],
                copies,
                axis=0,
            )

            self.data[key] = np.concatenate(
                [
                    self.data[key][:idx],
                    repeated,
                    self.data[key][idx:],
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

    def _on_close(self, event):
        with h5py.File(self.save_path, "w") as f:
            for key, value in self.data.items():
                f.create_dataset(
                    key,
                    data=value,
                    compression="gzip",
                )
            f.close()
        self.file.close()

    def run(self):
        self.build_ui()
        self.on_frame_change(0)
        plt.show()

    def _on_key_press(self, event):
        if event.key in self.key_pressed:
            return  # ignore key repeat

        self.key_pressed.add(event.key)

        if event.key == "right":
            value = min(int(self.frame_slider.val) + 1, int(self.frame_slider.valmax))
            self.frame_slider.set_val(value)
        elif event.key == "left":
            value = max(int(self.frame_slider.val) - 1, 0)
            self.frame_slider.set_val(value)

        elif event.key == "up":
            value = min(int(self.demo_slider.val) + 1, int(self.demo_slider.valmax))
            self.demo_slider.set_val(value)

        elif event.key == "down":
            value = max(int(self.demo_slider.val) - 1, 0)
            self.demo_slider.set_val(value)

    def _on_key_release(self, event):
        self.key_pressed.discard(event.key)
