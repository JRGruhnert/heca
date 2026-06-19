from dataclasses import dataclass
import random
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

from heca.agents.experts.expert import ExpertAgent
from heca.environment.scenes.scene import Scene
from heca.misc.base import Configurable


class TapasDemoSelector(Configurable):
    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        agent: ExpertAgent.Config
        dataset_name: str = "visual-scene-play-v0.h5"
        file_name: str = "demos.h5"
        random_ep: bool = True

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.load_path = (
            Scene.resolve_base(cfg.agent.scene) / "demos" / cfg.dataset_name
        )
        self.save_path = ExpertAgent.resolve(cfg.agent) / cfg.file_name
        self.train_dataset = h5py.File(self.load_path, "r")
        self.demos_file = h5py.File(self.save_path, "w")
        self.observations = self.train_dataset["rgb"]  # type: ignore
        self.actions = self.train_dataset["actions"]  # type: ignore
        self.terminals = np.asarray(self.train_dataset["terminals"])  # type: ignore
        self.ep_slices, self.ep_lengths, self.total_episodes = self._make_episode_meta()

        self.start_idx = 0
        self.ep_idx = 0
        self.key_pressed = set()

        self.demo_counter = 0
        self.out_datasets = {}

        self.btn_start_text = "Mark Start"
        self.btn_end_text = "Mark End & Add"
        self.btn_save_text = "Save & Close"

    def _make_episode_meta(self) -> tuple[list[slice], list[int], int]:
        terminal_idxs = np.where(self.terminals == 1.0)[0]
        slices, start = [], 0
        demo_lengths = []
        for end in terminal_idxs:
            slices.append(slice(start, end + 1))
            demo_lengths.append(end - start + 1)
            start = end + 1
        return (slices, demo_lengths, len(slices))

    def _build_ui(self):
        self.fig = plt.figure(figsize=(14, 8))
        self.fig.canvas.manager.set_window_title(f"Demo Selector: {self.cfg.agent.folder}")  # type: ignore

        # Image panel (right)
        self.ax_img = self.fig.add_axes((0.45, 0.08, 0.52, 0.88))
        self.ax_img.axis("off")

        # demos text
        self.demos_text = self.fig.text(
            0.23,
            0.95,
            "No demos saved yet.",
            fontsize=14,
            ha="center",
            va="top",
            family="monospace",
        )

        # Episode slider
        ax_ep = self.fig.add_axes((0.08, 0.88, 0.30, 0.04))
        self.ep_slider = Slider(
            ax=ax_ep,
            label="Episode",
            valmin=0,
            valmax=max(1, len(self.ep_slices) - 1),
            valinit=self.ep_idx,
            valstep=1,
        )

        # Frame slider
        ax_frame = self.fig.add_axes((0.08, 0.78, 0.30, 0.04))
        self.frame_slider = Slider(
            ax=ax_frame,
            label="Frame",
            valmin=0,
            valmax=max(1, self.ep_lengths[self.ep_idx]),
            valinit=self.start_idx,
            valstep=1,
        )

        # Buttons
        ax_bstart = self.fig.add_axes((0.05, 0.65, 0.16, 0.07))
        self.btn_start = Button(
            ax_bstart,
            self.btn_start_text,
            color="lightblue",
        )

        ax_bend = self.fig.add_axes((0.22, 0.65, 0.16, 0.07))
        self.btn_end = Button(
            ax_bend,
            self.btn_end_text,
            color="lightsalmon",
        )

        # Status text
        self.status_text = self.fig.text(
            0.05,
            0.60,
            "",
            fontsize=14,
            va="top",
        )
        self._update_to_default_status()

        self.ep_slider.on_changed(lambda v: self._on_episode_change(int(v)))
        self.frame_slider.on_changed(lambda v: self._on_frame_change(int(v)))
        self.btn_start.on_clicked(self.on_mark_start)
        self.btn_end.on_clicked(self.on_mark_end)

        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.fig.canvas.mpl_connect("key_release_event", self._on_key_release)
        self.fig.canvas.mpl_connect("close_event", self._on_close)

    def _update_demos_status(self):
        if self.demo_counter == 0:
            self.demos_text.set_text("No demos saved yet.")
        else:
            self.demos_text.set_text(f"{self.demo_counter} demo(s) saved.")
        self.fig.canvas.draw_idle()

    def _update_to_default_status(self):
        self.status_text.set_text(
            f"Current start frame {self.start_idx}.\n\n"
            f"Pick a new start frame and press: '{self.btn_start_text}'\n\n"
            f"or pick the end frame and press:  '{self.btn_end_text}'"
        )
        self.fig.canvas.draw_idle()

    def _update_to_idx_error_status(self, end_idx: int):
        self.status_text.set_text(
            "⚠ End frame must be after start frame!\n"
            f"Start frame is {self.start_idx} and end frame is {end_idx}"
        )
        self.fig.canvas.draw_idle()

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
            value = min(int(self.ep_slider.val) + 1, int(self.ep_slider.valmax))
            self.ep_slider.set_val(value)

        elif event.key == "down":
            value = max(int(self.ep_slider.val) - 1, 0)
            self.ep_slider.set_val(value)

    def _on_key_release(self, event):
        self.key_pressed.discard(event.key)

    def _on_episode_change(self, ep_idx: int):
        self.ep_idx = ep_idx
        self.start_idx = 0
        ep_len = self.ep_lengths[ep_idx]  # type: ignore

        self.frame_slider.valmax = ep_len - 1
        self.frame_slider.ax.set_xlim(0, ep_len - 1)
        self.frame_slider.set_val(0)
        self._update_to_default_status()
        self._on_frame_change(self.start_idx)

    def _on_frame_change(self, frame_idx: int):
        ep_slice = self.ep_slices[self.ep_idx]
        frame = self.observations[ep_slice][frame_idx]  # type: ignore
        self.ax_img.clear()
        self.ax_img.imshow(frame)  # type: ignore
        self.fig.canvas.draw_idle()

    def _on_close(self, event):
        self.train_dataset.close()
        self.demos_file.close()

    def on_mark_start(self, _event):
        self.start_idx = int(self.frame_slider.val)
        self._update_to_default_status()

    def on_mark_end(self, _event):
        end_idx = int(self.frame_slider.val)
        if end_idx <= self.start_idx:
            self._update_to_idx_error_status(end_idx)
            return
        self.save_demo(start=self.start_idx, end=end_idx, episode=self.ep_idx)
        self._prepare_new_episode()
        self._update_demos_status()
        self._update_to_default_status()

    def _prepare_new_episode(self):
        if self.cfg.random_ep:
            ep_idx = random.randrange(self.total_episodes - 1)
        else:
            old_ep_idx = self.ep_idx
            if old_ep_idx == len(self.ep_slices) - 1:
                ep_idx = 0
            else:
                ep_idx = old_ep_idx + 1

        self.ep_slider.set_val(ep_idx)

    def save_demo(self, start: int, end: int, episode: int):
        sl = self.ep_slices[episode]
        length = end - start + 1

        for key in self.train_dataset.keys():
            data = self.train_dataset[key][sl][start : end + 1]  # type: ignore
            self._append_dataset(key, data)

        demo_ids = np.full(length, self.demo_counter, dtype=np.int32)
        self._append_dataset("demo", demo_ids)

        self.demo_counter += 1

    def _append_dataset(self, name, data):
        n = len(data)

        if name not in self.out_datasets:
            self.out_datasets[name] = self.demos_file.create_dataset(
                name,
                data=data,
                maxshape=(None,) + data.shape[1:],
                chunks=True,
            )
        else:
            ds = self.out_datasets[name]
            old_n = ds.shape[0]
            ds.resize(old_n + n, axis=0)
            ds[old_n : old_n + n] = data

    def run(self):
        self._build_ui()
        self._on_episode_change(0)
        plt.show()
