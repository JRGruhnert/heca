from functools import cached_property
from typing import Any, cast

import numpy as np
from dataclasses import dataclass

import torch

from heca.agents.agent import AgentFeedback
from heca.entities.entity import Entity, Mobility
from heca.environment.scenes.scene import Scene
from heca.misc.td import TDScene, TDSceneImages
import ogbench
from ogbench.manipspace.envs.scene_env import SceneEnv


class OGBenchScene(Scene):
    @dataclass(kw_only=True)
    class Config(Scene.Config):
        label: str = "ogbench"
        folder: str = "samples"
        id: str = "visual-scene-play-v0"
        mode: str = "task"  #
        ob_type: str = "states"  # states, pixels
        width: int = 256
        height: int = 256
        visualize_info: bool = True

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg

        self.env = cast(
            SceneEnv,
            ogbench.make_env_and_datasets(
                dataset_name=cfg.id,
                env_only=True,
                dataset_only=False,
                ob_type=cfg.ob_type,
                mode=cfg.mode,
                visualize_info=cfg.visualize_info,
                width=cfg.width,
                height=cfg.height,
            ),
        )

    def close(self):
        self.env.close()

    @cached_property
    def entities(self) -> list[Entity]:
        ents = [
            Entity.Config(
                label="drawer",
                states={"open", "closed"},
                mobility=Mobility.ARTICULATED,
            ),
            Entity.Config(
                label="window",
                states={"open", "closed"},
                mobility=Mobility.ARTICULATED,
            ),
            Entity.Config(
                label="left-button",
                states={"free", "locked"},
                mobility=Mobility.STATIC,
            ),
            Entity.Config(
                label="right-button",
                states={"free", "locked"},
                mobility=Mobility.STATIC,
            ),
            Entity.Config(
                label="red_block",
                states={"grabbed", "drawer", "floor", "mia"},
                mobility=Mobility.FREE,
            ),
        ]
        return [Entity.create(e) for e in ents]

    @cached_property
    def cursor(self) -> Entity:
        config = Entity.Config(
            label="cursor",
            states={"open", "closed"},
            mobility=Mobility.FREE,
        )
        return Entity.create(config)

    def heca_td(self, obs) -> TDScene:
        raise NotImplementedError()

    def to_td_scene_images(self, obs) -> TDSceneImages:
        raise NotImplementedError()

    def sample_image(self) -> np.ndarray:
        raise NotImplementedError()

    def to_internal(self, obs: Any, info: dict[str, Any]) -> Any:
        return obs

    def sample_task(
        self,
    ) -> tuple[
        tuple[TDScene, TDSceneImages],
        tuple[TDScene, TDSceneImages],
    ]:
        self.og_obs, self.info = self.env.reset()
        obs_img = self.env.render(camera="front_pixels", depth=True)
        renderer = self.env._renderer
        assert renderer is not None
        renderer._mjr_context

    def _step(self, action: np.ndarray) -> Any:
        self.og_obs, reward, terminated, truncated, info = self.env.step(action)

    def get_cursor(self, obs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError()


# encoder Fix
# Dino Encoder Fix (padding)
# Encoder loading refs
# Image selector fix
