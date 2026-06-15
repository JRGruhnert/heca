from dataclasses import dataclass
import math
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np

from heca.agents.experts.expert import ExpertAgent
from heca.agents.experts.tapas import TapasAgent
from heca.environment.scenes.scene import Scene
from heca.misc.base import Configurable


class AgentTester(Configurable):
    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        agents: list[ExpertAgent.Config]
        scene: Scene.Config
        frame_time: float = 0.05

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.scene = Scene.get(self.cfg.scene)
        self.agents = [ExpertAgent.get(cfg) for cfg in cfg.agents]
        assert all(
            agent.cfg.scene == self.cfg.scene for agent in self.agents
        ), "Every agent must use the same scene config as cfg.scene."

    def _build_ui(self):
        self.fig = plt.figure(figsize=(14, 8))
        self.fig.canvas.manager.set_window_title(f"Agent Selector")  # type: ignore

        # Image panel (right)
        self.ax_img = self.fig.add_axes((0.45, 0.08, 0.52, 0.88))
        self.ax_img.axis("off")

        self.buttons = []

        n_cols = 4
        n_agents = len(self.agents)
        n_rows = math.ceil(n_agents / n_cols)

        # Layout parameters
        x0 = 0.05
        y0 = 0.80
        btn_w = 0.10
        btn_h = 0.10
        x_gap = 0.02
        y_gap = 0.02

        for i, agent in enumerate(self.agents):
            row = i // n_cols
            col = i % n_cols

            x = x0 + col * (btn_w + x_gap)
            y = y0 - row * (btn_h + y_gap)

            ax = self.fig.add_axes((x, y, btn_w, btn_h))

            button = Button(
                ax,
                str(agent.cfg.folder),  # or whatever label you want
                color="lightgray",
            )

            # capture current agent
            button.on_clicked(lambda event, agent=agent: self.on_agent_selected(agent))

            self.buttons.append(button)

    def on_agent_selected(self, agent: ExpertAgent):
        # self.x = agent.act(self.x, self.y)
        assert isinstance(agent, TapasAgent), "Currently only supports TapasAgent"
        agent.policy.reset_episode()
        print("A")
        xt = agent.tapas_td(self.x, self.y)
        print("B")
        predictions = agent.make_prediction(xt)
        print("C")
        if predictions is None:
            raise NotImplementedError()
        print("D")
        while not predictions.is_finished:
            pred = predictions.step()
            action = np.concatenate((pred.ee, pred.gripper))  # type: ignore
            self.x, npimage = self.scene.step_vis(action)
            self.img_artist.set_data(npimage)
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            plt.pause(self.cfg.frame_time)
        print("E")

    def reset(self):
        print("Z")
        (self.x, x_image), (self.y, _) = self.scene.sample_task_vis()
        print("Y")
        self.ax_img.clear()
        self.img_artist = self.ax_img.imshow(x_image)
        self.ax_img.axis("off")
        self.fig.canvas.draw_idle()
        print("X")

    def run(self):
        self._build_ui()
        self.reset()
        plt.show()
