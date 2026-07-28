from dataclasses import dataclass
from textwrap import dedent

import numpy as np

from heca.misc.data import DCEntity, DCScene
from heca.misc.entity import Entity, Mobility
from heca.scenes.ogbench.scene import OGBenchScene
from heca.scenes.ogbench.utils import eval_pos, eval_state


class OGBenchScene5(OGBenchScene):
    @dataclass(kw_only=True)
    class Config(OGBenchScene.Config):
        label: str = "ogbench"
        tag: str = "samples"
        id: str = "visual-scene-play-v0"
        mode: str = "task"
        ob_type: str = "pixels"  # states, pixels
        visualize_info: bool = False

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg

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
    def entities(self) -> set[Entity]:
        ents = [
            Entity.Config(
                label="drawer_handle",
                scene="ogbench",
                question="What describes the drawer the best?",
                answers=["It is open", "It is closed"],
                states=["open", "closed"],  # 0, 1
                mobility=Mobility.ARTICULATED,
                eval_func=eval_pos,
            ),
            Entity.Config(
                label="window_handle",
                scene="ogbench",
                question="What describes the sliding window the best?",
                answers=[
                    "it is open and therefore moved to the front",
                    "it is closed and therefore moved to the back",
                ],
                states=["open", "closed"],  # 0, 1
                mobility=Mobility.ARTICULATED,
                eval_func=eval_pos,
            ),
            Entity.Config(
                label="button_0",
                scene="ogbench",
                question="What is the color of the left button?",
                answers=["white", "red"],
                states=["free", "locked"],  # 0, 1
                mobility=Mobility.STATIC,
                eval_func=eval_state,
            ),
            Entity.Config(
                label="button_1",
                scene="ogbench",
                question="What is the color of the right button?",
                answers=["white", "red"],
                states=["free", "locked"],  # 0, 1
                mobility=Mobility.STATIC,
                eval_func=eval_state,
            ),
            Entity.Config(
                label="block_0",
                scene="ogbench",
                question="Where is the red cube in the scene?",
                answers=[
                    "inside the drawer",
                    "on the floor",
                    "unknown, cause it is not visible",
                ],
                states=["drawer", "floor", "unknown"],  # 0, 1
                mobility=Mobility.FREE,
                eval_func=eval_pos,
            ),
        ]
        return set([Entity.get(e) for e in ents])

    @property
    def ee(self) -> Entity:
        config = Entity.Config(
            label="ee",
            scene="ogbench",
            question="Where is the red cube in the scene?",
            answers=[
                "on the floor",
                "inside the drawer",
                "unknown, cause it is not visible",
            ],
            states=["open", "closed"],  # 0, 1
            mobility=Mobility.FREE,
        )
        return Entity.get(config)

    def to_dc_scene(self, obs: dict) -> DCScene:
        dc_entities: dict[str, DCEntity] = {}
        for entity in self.entities:
            if entity.cfg.label in [
                "button_0",
                "button_1",
            ]:  # hack cause _pos already used
                e_pos = obs[f"privileged_{entity.cfg.label}_pos_full"]
            else:
                e_pos = obs[f"privileged_{entity.cfg.label}_pos"]
            wxyz = obs[f"privileged_{entity.cfg.label}_quat"]
            e_rot = np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]], dtype=np.float32)
            e_ste = np.atleast_1d(obs[f"privileged_{entity.cfg.label}_state"])
            e_soh = entity.one_hot_from_idx_dc(e_ste)
            dc_entities[entity.cfg.label] = Entity.to_value(e_pos, e_rot, e_ste, e_soh)
        pos, rot, ste, soh = self.get_ee_dc(obs)
        ee = Entity.to_value(pos, rot, ste, soh)
        extras = self.get_extras(obs)
        return DCScene(ee, dc_entities, extras=extras)
