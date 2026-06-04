from typing import Any, cast

import numpy as np
from dataclasses import dataclass

import torch
import gymnasium as gym

from heca.agents.agent import AgentFeedback
from heca.entities.entity import Entity
from heca.environment.scenes.scene import Scene
from heca.misc.td import TDScene, TDSceneImages
from ogbench import manipspace
import ogbench


class OGBenchScene(Scene):
    @dataclass(kw_only=True)
    class Config(Scene.Config):
        label: str = "ogbench"
        folder: str = "samples"
        id: str = "visual-scene-play-v0"
        mode: str = "task"  #
        ob_type: str = "pixels"  # states, pixels
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

        self.env = cast(
            gym.Env,
            ogbench.make_env_and_datasets(
                dataset_name=cfg.id,
                env_only=True,
                dataset_only=False,
                ob_type=cfg.ob_type,
                mode=cfg.mode,
                terminate_at_goal=cfg.terminate_at_goal,
                success_timing=cfg.success_timing,
                physics_timestep=cfg.physics_timestep,
                control_timestep=cfg.control_timestep,
                visualize_info=cfg.visualize_info,
                width=cfg.width,
                height=cfg.height,
            ),
        )

    def close(self):
        self.env.close()

    @property
    def entities(self) -> list[Entity]:
        return []

    @property
    def cursor(self) -> Entity:
        raise NotImplementedError()

    def heca_td(self, obs) -> TDScene:
        raise NotImplementedError()

    def image_tensors(self, obs) -> dict[str, torch.Tensor]:
        raise NotImplementedError()

    def to_td_scene_images(self, obs) -> TDSceneImages:
        raise NotImplementedError()

    def image_numpy(self, obs) -> dict[str, np.ndarray]:
        raise NotImplementedError()

    def _reset(self) -> Any:
        self.og_obs, self.info = self.env.reset()

    def _step(self, action: np.ndarray) -> Any:
        self.og_obs, reward, done, end, info = self.env.step(action)

    def test(self):
        obs, info = self.env.reset()
        # print(obs.keys())
        print(info.keys())
        print(info["goal"])
        frame = self.env.render()

        action = self.env.action_space.sample()
        print(action)

    def get_cursor(self, obs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError()
