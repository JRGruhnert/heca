from dataclasses import dataclass

import numpy as np
import pickle
from pathlib import Path
import h5py
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

from heca.misc.base import Configurable


class DemoSelector(Configurable):
    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        demos_name: str
        dataset_name: str
        base_path: Path = Path("data/demos")
        scene_name: str = "ogbench"
        file_name: str = "demos.pkl"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.save_path = cfg.base_path / cfg.scene_name / cfg.demos_name / cfg.file_name
        self.dataset_path = cfg.base_path / cfg.scene_name / "full" / cfg.dataset_name
        print(self.dataset_path.resolve())
        self.train_dataset = h5py.File(self.dataset_path, "r")
        self.observations = self.train_dataset["observations"]  # type: ignore
        self.actions = self.train_dataset["actions"]  # type: ignore
        self.terminals = np.asarray(self.train_dataset["terminals"])  # type: ignore

        self.episode_slices = self._make_episode_slices()

        self.saved_segments: list[dict] = []
        self.pending_start: int | None = None

    def _make_episode_slices(self) -> list[slice]:
        terminal_idxs = np.where(self.terminals == 1.0)[0]
        slices, start = [], 0
        for end in terminal_idxs:
            slices.append(slice(start, end + 1))
            start = end + 1
        return slices

    def _build_ui(self):
        self.fig = plt.figure(figsize=(14, 8))
        self.fig.canvas.manager.set_window_title("Demo Selector")  # type: ignore

        # Image panel (right)
        self.ax_img = self.fig.add_axes((0.45, 0.08, 0.52, 0.88))
        self.ax_img.axis("off")

        # Episode slider
        ax_ep = self.fig.add_axes((0.08, 0.88, 0.30, 0.04))
        self.ep_slider = Slider(
            ax_ep,
            "Episode",
            0,
            max(1, len(self.episode_slices) - 1),
            valinit=0,
            valstep=1,
        )

        # Frame slider
        ax_frame = self.fig.add_axes((0.08, 0.78, 0.30, 0.04))
        self.frame_slider = Slider(ax_frame, "Frame", 0, 1, valinit=0, valstep=1)

        # Buttons
        ax_bstart = self.fig.add_axes((0.05, 0.65, 0.13, 0.07))
        self.btn_start = Button(ax_bstart, "Mark Start", color="lightblue")

        ax_bend = self.fig.add_axes((0.22, 0.65, 0.17, 0.07))
        self.btn_end = Button(ax_bend, "Mark End & Add", color="lightgreen")

        ax_bclear = self.fig.add_axes((0.05, 0.55, 0.13, 0.07))
        self.btn_clear = Button(ax_bclear, "Clear Pending", color="lightyellow")

        ax_bsave = self.fig.add_axes((0.22, 0.55, 0.17, 0.07))
        self.btn_save = Button(ax_bsave, "Save Segments", color="lightsalmon")

        # Status and segments text
        self.status_text = self.fig.text(0.05, 0.48, "", fontsize=9, va="top")
        self.segments_text = self.fig.text(
            0.05,
            0.42,
            "No segments selected yet.",
            fontsize=8,
            va="top",
            family="monospace",
        )

        self.ep_slider.on_changed(lambda v: self._on_episode_change(int(v)))
        self.frame_slider.on_changed(lambda v: self._render_frame(int(v)))
        self.btn_start.on_clicked(self.on_mark_start)
        self.btn_end.on_clicked(self.on_mark_end)
        self.btn_clear.on_clicked(self.on_clear)
        self.btn_save.on_clicked(self.on_save)

        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)

    def _on_key_press(self, event):
        if event.key in ("right"):
            value = min(int(self.frame_slider.val) + 1, int(self.frame_slider.valmax))
            self.frame_slider.set_val(value)
        elif event.key in ("left"):
            value = max(int(self.frame_slider.val) - 1, 0)
            self.frame_slider.set_val(value)
        elif event.key in ("up"):
            value = min(int(self.ep_slider.val) + 1, int(self.ep_slider.valmax))
            self.ep_slider.set_val(value)
        elif event.key in ("down"):
            value = max(int(self.ep_slider.val) - 1, 0)
            self.ep_slider.set_val(value)

    def run(self):
        self._build_ui()
        self._on_episode_change(0)
        plt.show()

    def _on_episode_change(self, ep_idx: int):
        self.pending_start = None
        sl = self.episode_slices[ep_idx]
        ep_len = len(self.observations[sl])  # type: ignore
        self.frame_slider.valmax = ep_len - 1
        self.frame_slider.ax.set_xlim(0, ep_len - 1)
        self.frame_slider.set_val(0)
        self.status_text.set_text(f"Episode {ep_idx}: {ep_len} frames")
        self._render_frame(0)

    def _render_frame(self, frame_idx: int):
        ep_idx = int(self.ep_slider.val)
        sl = self.episode_slices[ep_idx]
        ep_obs = self.observations[sl]  # type: ignore
        ep_len = len(ep_obs)  # type: ignore

        self.ax_img.clear()
        self.ax_img.imshow(ep_obs[frame_idx])  # type: ignore
        title = f"Episode {ep_idx} | Frame {frame_idx}/{ep_len - 1}"
        if self.pending_start is not None:
            title += f" | Start: {self.pending_start}"
            color = "green" if frame_idx >= self.pending_start else "red"
            for spine in self.ax_img.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(4)
        self.ax_img.set_title(title)
        self.ax_img.axis("off")
        self.fig.canvas.draw_idle()

    def _update_segments_view(self):
        if not self.saved_segments:
            self.segments_text.set_text("No segments selected yet.")
        else:
            lines = [f"{len(self.saved_segments)} segment(s):"]
            for i, seg in enumerate(self.saved_segments):
                lines.append(
                    f"[{i}] ep={seg['episode']}  frames={seg['start']}–{seg['end']}"
                    f"  ({seg['end'] - seg['start'] + 1} steps)"
                )
            self.segments_text.set_text("\n".join(lines))
        self.fig.canvas.draw_idle()

    def on_mark_start(self, _event):
        self.pending_start = int(self.frame_slider.val)
        self.status_text.set_text(
            f"Start marked at frame {self.pending_start}. Pick end frame and click 'Mark End & Add'."
        )
        self._render_frame(self.pending_start)

    def on_mark_end(self, _event):
        if self.pending_start is None:
            self.status_text.set_text("⚠ Mark a start frame first!")
            self.fig.canvas.draw_idle()
            return
        end = int(self.frame_slider.val)
        if end <= self.pending_start:
            self.status_text.set_text("⚠ End frame must be after start frame!")
            self.fig.canvas.draw_idle()
            return
        ep_idx = int(self.ep_slider.val)
        sl = self.episode_slices[ep_idx]
        self.saved_segments.append(
            {
                "episode": ep_idx,
                "start": self.pending_start,
                "end": end,
                "observations": self.observations[sl][  # type: ignore
                    self.pending_start : end + 1
                ].copy(),  # type: ignore
                "actions": self.actions[sl][self.pending_start : end + 1].copy(),  # type: ignore
            }
        )
        self.status_text.set_text(
            f"✓ Segment added (ep={ep_idx}, frames {self.pending_start}–{end}). "
            f"Total: {len(self.saved_segments)}"
        )
        self.pending_start = None
        self._update_segments_view()
        self._render_frame(int(self.frame_slider.val))

    def on_clear(self, _event):
        self.pending_start = None
        self.status_text.set_text("Pending start cleared.")
        self._render_frame(int(self.frame_slider.val))

    def on_save(self, _event):
        if not self.saved_segments:
            self.status_text.set_text("⚠ No segments to save!")
            self.fig.canvas.draw_idle()
            return
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.save_path, "wb") as f:
            pickle.dump(self.saved_segments, f)
        self.status_text.set_text(
            f"✓ Saved {len(self.saved_segments)} segments to {self.save_path}"
        )
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def load_segments(self) -> list[dict]:
        with open(self.cfg.base_path, "rb") as f:
            loaded = pickle.load(f)
        print(f"{len(loaded)} segments loaded:")
        for i, seg in enumerate(loaded):
            print(
                f"  [{i}] episode={seg['episode']}  frames={seg['start']}–{seg['end']}"
                f"  obs={seg['observations'].shape}  actions={seg['actions'].shape}"
            )
        return loaded
