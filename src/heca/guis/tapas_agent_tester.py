from dataclasses import dataclass
import atexit

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np

from heca.experts.expert import ExpertModel
from heca.experts.tapas import TapasExpert
from heca.scenes.ogbench.scene import OGScene
from heca.misc import logger
from heca.misc.base import Configurable


class TapasManualExecuter(Configurable):
    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        agents: list[ExpertModel.Config]
        scene: OGScene.Config
        frame_time: float = 0.05
        use_gt: bool = True

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.scene = OGScene.get(self.cfg.scene, auto_load=False)
        self.agents: list[ExpertModel] = []
        for agent_cfg in cfg.agents:
            agent = ExpertModel.get(agent_cfg, auto_load=False)
            agent.use_gt(self.cfg.use_gt)
            agent.load()
            self.agents.append(agent)
        self._viewer_launched = False
        atexit.register(self._close_viewer)
        assert all(
            agent.cfg.scene == self.cfg.scene for agent in self.agents
        ), "Every agent must use the same scene config as cfg.scene."

    def _build_ui(self):
        self.fig = plt.figure(figsize=(8, 6))
        self.fig.canvas.manager.set_window_title("Expert Agent Tester")  # type: ignore
        self.fig.canvas.mpl_connect("close_event", lambda event: self._close_viewer())

        self.buttons = []

        n_cols = 3
        btn_w = 0.14
        btn_h = 0.10
        x_gap = 0.02
        y_gap = 0.02

        # Center the button grid horizontally.
        grid_w = n_cols * btn_w + (n_cols - 1) * x_gap
        x0 = (1.0 - grid_w) / 2
        y0 = 0.72

        # Reset / Quit controls, centered above the grid.
        top_w = 2 * btn_w + x_gap
        top_x0 = (1.0 - top_w) / 2

        ax_extra = self.fig.add_axes((top_x0, y0 + btn_h + y_gap, btn_w, btn_h))
        self.extra_button = Button(ax_extra, "Reset", color="lightblue")
        self.extra_button.on_clicked(lambda event: self.reset())

        ax_quit = self.fig.add_axes(
            (top_x0 + btn_w + x_gap, y0 + btn_h + y_gap, btn_w, btn_h)
        )
        self.quit_button = Button(ax_quit, "Quit", color="lightcoral")
        self.quit_button.on_clicked(lambda event: self.close())

        for i, agent in enumerate(self.agents):
            row = i // n_cols
            col = i % n_cols

            x = x0 + col * (btn_w + x_gap)
            y = y0 - row * (btn_h + y_gap)

            ax = self.fig.add_axes((x, y, btn_w, btn_h))
            button = Button(ax, agent.cfg.tag, color="lightgray")
            button.on_clicked(lambda event, agent=agent: self.on_agent_selected(agent))
            self.buttons.append(button)

    def on_agent_selected(self, agent: ExpertModel):
        assert isinstance(agent, TapasExpert), "Currently only supports TapasAgent"
        agent.policy.reset_episode()
        xt = agent.tapas_td(self.x, self.y)
        if agent.cfg.policy.return_full_batch:
            predictions = agent.make_batch_prediction(xt)
            if predictions is None:
                raise NotImplementedError

            while not predictions.is_finished:
                pred = predictions.step()
                action = np.concatenate((pred.ee, pred.gripper))  # type: ignore
                tdscene, tdimage, _ = self.scene.step_vis(action)
                self.gui_step()
            self.x = agent.make_scene(tdscene, tdimage)
        else:
            while not (pred := agent.make_prediction(xt))[1]:
                action, _ = pred
                if action is None:
                    raise NotImplementedError
                tdscene, tdimage, _ = self.scene.step_vis(action)
                self.x = agent.make_scene(tdscene, tdimage)
                xt = agent.tapas_td(self.x, self.y)
                self.gui_step()

    def gui_step(self):
        self.scene.env.unwrapped.sync_passive_viewer()
        plt.pause(self.cfg.frame_time)

    def reset(self):
        (self.x, _, _), (self.y, _, _) = self.scene.sample_task_vis()
        if not self._viewer_launched:
            self.scene.env.unwrapped.launch_passive_viewer()
            self._viewer_launched = True
        self.scene.env.unwrapped.sync_passive_viewer()
        self.fig.canvas.draw_idle()

    def _close_viewer(self):
        if not self._viewer_launched:
            return
        try:
            self.scene.env.unwrapped.close_passive_viewer()
        except Exception as e:
            logger.warning(f"Failed to close passive viewer: {e}")
        finally:
            self._viewer_launched = False

    def close(self):
        self._close_viewer()
        plt.close(self.fig)

    def run(self):
        self._build_ui()
        self.reset()
        try:
            plt.show()
        finally:
            self._close_viewer()
