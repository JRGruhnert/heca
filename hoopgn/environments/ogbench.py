import numpy as np
from dataclasses import dataclass

from hoopgn.environments.environment import Environment, StepFeedback
from hoopgn.observation.td_properties import TDProperties
import gymnasium as gym
from ogbench import manipspace


class OGBenchEnvironment(Environment):
    @dataclass(kw_only=True)
    class Config(Environment.Config):
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

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg

        self.env = gym.make(
            id=cfg.id,
            ob_type=cfg.ob_type,
            mode=cfg.mode,
            terminate_at_goal=cfg.terminate_at_goal,
            success_timing=cfg.success_timing,
            physics_timestep=cfg.physics_timestep,
            control_timestep=cfg.control_timestep,
            visualize_info=cfg.visualize_info,
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

    def get_observation(self) -> TDProperties:
        state_dict = {}
        # TODO: Implement proper observation parsing
        return TDProperties(state_dict)
