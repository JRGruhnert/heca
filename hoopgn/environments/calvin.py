import numpy as np
from dataclasses import dataclass, field

import torch

from hoopgn.converters.calvin_hoop import CalvinHoopConverter
from hoopgn.converters.calvin_tapas import CalvinTapasConverter
from hoopgn.converters.converter import HoopConverter, LeafConverter
from hoopgn.environments.environment import Environment

from tapas_gmm_modified.env.calvin import Calvin, CalvinConfig

from hoopgn.misc.area import Area
from hoopgn.misc.state import State
from hoopgn.misc.td import TDScene


@dataclass(kw_only=True)
class CalvinAreaConfig(Area.Config):
    labels: set[str] = field(
        default_factory=lambda: {"table", "drawer_open", "drawer_closed"}
    )
    spawn_surfaces: dict = field(
        default_factory=lambda: {
            "table": [[0.0, -0.15, 0.46], [0.30, -0.03, 0.46]],
            "drawer_open": [[0.04, -0.35, 0.38], [0.30, -0.21, 0.38]],
            "drawer_closed": [[0.04, -0.16, 0.38], [0.30, -0.03, 0.38]],
        }
    )
    eval_surfaces: dict = field(
        default_factory=lambda: {
            "table": [[-0.02, -0.17, 0.44], [0.32, -0.01, 0.54]],
            "drawer_open": [[0.02, -0.37, 0.34], [0.32, -0.23, 0.44]],
            "drawer_closed": [[0.02, -0.18, 0.34], [0.32, -0.00, 0.44]],
        }
    )


class CalvinEnvironment(Environment):
    @dataclass(kw_only=True)
    class Query(Environment.Query):
        label: str = "calvin"

    @dataclass(kw_only=True)
    class Config(Environment.Config):
        hoop_cv: HoopConverter.Config = CalvinHoopConverter.Config()
        leaf_cv: LeafConverter.Config = CalvinTapasConverter.Config()

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
        self.state = State.from_area_config(
            CalvinAreaConfig(),
        )
        self.overrides = set(
            [
                "block_red_position",
                "block_blue_position",
                "block_pink_position",
            ]
        )

    def close(self):
        self.env.close()

    def reset(self):
        obs = self.env.reset()[0]
        x = self.to_observation(obs)
        return self.modify(x)

    def step(self, action: np.ndarray) -> TDScene:
        obs = self.env.step(action, render=False)[0]
        return self.to_observation(obs)

    def modify(self, x: TDScene) -> TDScene:
        v1 = x.data["scene"]["v1"]
        for k in self.overrides:
            v1[k] = torch.cat([v1[k], self.state(v1[k])])
        return x

    def validate(self, x: TDScene) -> bool:
        v1 = x.data["scene"]["v1"]
        for k in self.overrides:
            if torch.all(v1[k][3:] == 0):
                return False
        return True

    def sample(self) -> TDScene:
        x = self.reset()
        while not self.validate(x):
            x = self.reset()
        return x

    def render(self):
        raise NotImplementedError("Render method not implemented yet.")
