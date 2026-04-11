import numpy as np
from dataclasses import dataclass

from tapas_gmm_modified.env.calvin import Calvin, CalvinConfig
from src.environments.environment import Environment, EnvironmentConfig, StepFeedback
from src.observation.observation import StateValueDict
from src.observation.calvin import CalvinObservation


@dataclass(kw_only=True)
class CalvinEnvironmentConfig(EnvironmentConfig):
    calvin_config: CalvinConfig = CalvinConfig(
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


class CalvinEnvironment(Environment):
    def __init__(self, config: CalvinEnvironmentConfig):
        self.config = config

        self.env = Calvin(self.config.calvin_config)

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

    def get_observation(self) -> StateValueDict:
        return CalvinObservation.from_internal(
            self.calvin_obs, converters=self.converters
        )

    def close(self):
        self.env.close()
