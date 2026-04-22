from typing import Any

import numpy as np
from dataclasses import dataclass

from hoopgn.environments.environment import Environment

from tapas_gmm_modified.env.calvin import Calvin, CalvinConfig


class CalvinEnvironment(Environment):
    @dataclass(kw_only=True)
    class Signature(Environment.Signature):
        label: str = "calvin"

    @dataclass(kw_only=True)
    class Config(Environment.Config):
        cc: CalvinConfig = CalvinConfig(
            task="Undefined",
            cameras=("wrist", "front"),
            camera_pose={},
            image_size=(256, 256),
            static=False,
            headless=False,
            scale_action=False,
            delay_gripper=False,
            gripper_plot=False,
            postprocess_actions=False,
            eval_mode=False,
            real_time=False,
            pybullet_vis=False,
        )

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.env = Calvin(self.cfg.cc)

    def close(self):
        self.env.close()

    def sample(self) -> Any:
        return self.env.reset()[0]

    def step(self, action: np.ndarray) -> Any:
        return self.env.step(action, render=False)[0]

    def render(self):
        raise NotImplementedError("Render method not implemented yet.")
