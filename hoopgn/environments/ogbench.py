import numpy as np
from dataclasses import dataclass

from hoopgn.environments.environment import Environment, EnvironmentConfig, StepFeedback
from hoopgn.observation.td_parameters import TDParameters, empty_batchsize
import gymnasium as gym
from ogbench import manipspace


@dataclass(kw_only=True)
class OGBenchEnvironmentConfig(EnvironmentConfig):
    id: str = "cube-single-v0"
    mode: str = "task"  #
    ob_type: str = "states"  # states, pixels
    width: int = 256
    height: int = 256
    terminate_at_goal: bool = True
    success_timing: str = "post"  # pre, post
    physics_timestep: float = 0.002
    control_timestep: float = 0.05
    visualize_info: bool = True


class OGBenchEnvironment(Environment):
    def __init__(self, config: OGBenchEnvironmentConfig):
        super().__init__(config)
        self.config = config

        self.env = gym.make(
            id=config.id,
            ob_type=config.ob_type,
            mode=config.mode,
            terminate_at_goal=config.terminate_at_goal,
            success_timing=config.success_timing,
            physics_timestep=config.physics_timestep,
            control_timestep=config.control_timestep,
            visualize_info=config.visualize_info,
        )

    def close(self):
        self.env.close()

    def _reset(self):
        self.og_obs, self.info = self.env.reset()

    def step(
        self,
        action: np.ndarray,
    ) -> StepFeedback:
        self.og_obs = self.env.step(action)
        # TODO: In Future implement error detection and return other Codes
        return StepFeedback.OKAY

    def render(self):
        raise NotImplementedError("Render method not implemented yet.")

    def get_observation(self) -> TDParameters:
        state_dict = {}
        # TODO: Implement proper observation parsing
        return TDParameters(state_dict, batch_size=empty_batchsize)
