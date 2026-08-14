from dataclasses import dataclass
from textwrap import dedent
from typing import cast

import numpy as np
import ogbench

from heca.data.free import FreeEntity
from heca.data.prismatic import PrismaticEntity
from heca.data.static import StaticEntity
from heca.scenes.ogbench.scene import OGScene
from heca.data.data import DCEntity, DCScene
from heca.data.entity import Entity
from heca.utils.quaternion import Quaternion
from ogbench.manipspace.envs.scene_env import ManipSpaceEnv


class OGSceneOG(OGScene):
    @dataclass(kw_only=True)
    class Config(OGScene.Config):
        tag: str = "scene0"
        vis: bool = True

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.env = cast(
            ManipSpaceEnv,
            ogbench.make_env_and_datasets(
                dataset_name="visual-scene-play-v0",
                env_only=True,
                dataset_only=False,
                control_timestep=0.5,
            ),
        )

    @property
    def description(self) -> str:
        return dedent("""
            Scene Description

            The image shows a virtual robot manipulation environment.

            Fixed properties:
            - The robot arm is always semi-transparent and purple.
            - The sliding window is always located on the right side of the scene.
            - The drawer is always located on the left side of the scene.
            - Two buttons are always located near the front-center of the scene:
            - a left button,
            - a right button.

            Objects:
            - A semi-transparent purple robot arm.
            - A white sliding window with a white handle.
                The window can be opened by pulling it toward the camera.
            - A white drawer with a white handle.
            - A red cube.

            Variable properties:
            - The window may be open or closed.
            - The drawer may be open or closed.
            - The red cube may be:
                - on the floor,
                - inside the drawer, or
                - outside the camera's field of view.
            - Each button may be either red or white.

            Button color indicates state:
            - Red = locked
            - White = unlocked
        """)

    @property
    def entities(self) -> dict[str, Entity]:
        ents = {
            "drawer_handle": PrismaticEntity.Config(
                question="What describes the drawer the best?",
                answers=["It is open", "It is closed"],
                # states=["open", "closed"],  # 0, 1
            ),
            "window_handle": PrismaticEntity.Config(
                question="What describes the sliding window the best?",
                answers=[
                    "it is open and therefore moved to the front",
                    "it is closed and therefore moved to the back",
                ],
                # states=["open", "closed"],  # 0, 1
            ),
            "button_0": StaticEntity.Config(
                question="What is the color of the left button?",
                answers=["white", "red"],
                # states=["free", "locked"],  # 0, 1
                n_states=2,
            ),
            "button_1": StaticEntity.Config(
                question="What is the color of the right button?",
                answers=["white", "red"],
                # states=["free", "locked"],  # 0, 1
                n_states=2,
            ),
            "block_0": FreeEntity.Config(
                question="Where is the red cube in the scene?",
                answers=[
                    "inside the drawer",
                    "on the floor",
                    "unknown, cause it is not visible",
                ],
                # states=["drawer", "floor", "unknown"],  # 0, 1
            ),
        }
        return {l: Entity.get(e) for l, e in ents.items()}

    def to_dc_scene(self, obs: dict) -> DCScene:
        dc_entities: dict[str, DCEntity] = {}
        for label, entity in self.entities.items():
            if label in ["button_0", "button_1"]:  # hack
                e_pos = obs[f"privileged_{label}_pos_full"]
            else:
                e_pos = obs[f"privileged_{label}_pos"]
            wxyz = obs[f"privileged_{label}_quat"]
            e_rot = np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]], dtype=np.float32)
            e_ste = np.atleast_1d(obs[f"privileged_{label}_state"])
            value = np.concatenate((e_pos, e_rot, e_ste))
            feature = self.gnn_format(value, entity.cfg.n_states)
            dc_entities[label] = DCEntity(value=value, feature=feature)
        extras = self.get_extras(obs)
        return DCScene(dc_entities, extras=extras)

    def gnn_format(self, value: np.ndarray, n_states):
        # Initialize with zeros
        feat = np.zeros((56), dtype=np.float32)
        feat[0:3] = value[0:3]
        feat[3:6] = -10.0
        feat[6:10] = Quaternion.normalize(value[3:7])
        feat[10:13] = -10.0
        state_ids = value[7].astype(int)  # [N]
        feat[13 : 13 + n_states] = -10.0
        feat[13 + state_ids] = 10.0
        return feat
