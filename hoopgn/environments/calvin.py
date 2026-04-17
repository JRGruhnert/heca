import numpy as np
from dataclasses import dataclass

from tapas_gmm_modified.env.calvin import Calvin, CalvinConfig
import torch
from hoopgn.environments.environment import Environment, StepFeedback
from hoopgn.observation.td_properties import TDProperties


class CalvinEnvironment(Environment):
    @dataclass(kw_only=True)
    class Config(Environment.Config):
        label: str = "calvin"
        calvin: CalvinConfig = CalvinConfig(
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
        super().__init__(cfg)
        self.cfg = cfg
        self.env = Calvin(self.cfg.calvin)

    def close(self):
        self.env.close()

    def _reset(self):
        self.calvin_obs = self.env.reset(settle_time=50)[0]

    def step(
        self,
        action: np.ndarray,
    ) -> StepFeedback:
        self.calvin_obs = self.env.step(action, render=False)[0]
        # TODO: In Future implement error detection and return other Codes
        return StepFeedback.OKAY

    def render(self):
        raise NotImplementedError("Render method not implemented yet.")

    def get_observation(self) -> TDProperties:
        state_dict = {}
        state_dict["ee_position"] = torch.tensor(
            self.calvin_obs.ee_pose[:3], dtype=torch.float32
        )
        state_dict["ee_rotation"] = torch.tensor(
            self.calvin_obs.ee_pose[-4:], dtype=torch.float32
        )
        state_dict["ee_scalar"] = torch.tensor(
            np.array([self.calvin_obs.ee_state]), dtype=torch.float32
        )

        for label, value in self.calvin_obs.object_poses.items():
            k = label.removeprefix("base__")
            state_dict[f"{k}_position"] = torch.tensor(value[:3], dtype=torch.float32)
            state_dict[f"{k}_rotation"] = torch.tensor(value[-4:], dtype=torch.float32)

        for label, value in self.calvin_obs.object_states.items():
            k = label.removeprefix("base__")
            state_dict[f"{k}_scalar"] = torch.tensor(
                np.array([value]), dtype=torch.float32
            )

        for converter in self.converters:
            state_dict[converter.config.label] = converter(self.calvin_obs)

        return TDProperties(state_dict)
