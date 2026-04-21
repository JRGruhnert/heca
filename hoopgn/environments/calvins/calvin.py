import numpy as np
from dataclasses import dataclass

from hoopgn.environments.environment import Environment
from hoopgn.observation.td_scene import TDScene

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

    def _sample(self):
        self.calvin_obs = self.env.reset()[0]

    def _step(self, action: np.ndarray) -> TDScene:
        self.calvin_obs = self.env.step(action, render=False)[0]
        return self.observation()

    def render(self):
        raise NotImplementedError("Render method not implemented yet.")

    def observation(self) -> TDScene:
        # Prepare base fields
        scene_kwargs = {
            "v1": self._get_v1(),
            "v2": self._get_v2(),
        }
        # Add per-converter fields
        for converter in getattr(self, "converters", []):
            label = getattr(converter.cfg, "label", None)
            if label is not None:
                scene_kwargs[label] = converter(self.calvin_obs)
        return TDScene(**scene_kwargs)
