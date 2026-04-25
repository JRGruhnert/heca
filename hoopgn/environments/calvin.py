import numpy as np
from dataclasses import dataclass, field

from hoopgn.environments.environment import Environment

from tapas_gmm_modified.env.calvin import Calvin, CalvinConfig

from hoopgn.misc.area import Area
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
        self.area = Area(CalvinAreaConfig())

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
        x

    def validate(self, x: TDScene) -> bool:
        x["v1"]["area"] = self.area(x)
        ax = self.area.label(x)
        return ax is not None

    def is_valid(self, x: TDScene) -> bool:
        assert self.goal is not None, "Goal must be set before calling is_valid"
        for p in self.properties:
            l = p.cfg.label
            if not p.validate(x[l], self.goal[l]):
                return False
        return True

    def sample(self) -> TDScene:
        x = self.reset()
        while not self.validate(x):
            x = self.reset()
        return x

    def render(self):
        raise NotImplementedError("Render method not implemented yet.")
