from functools import cached_property

import torch
import numpy as np
from dataclasses import dataclass, field
from heca.misc.data import DCEntity, DCScene, TDImage
from heca.misc.entity import Entity, Mobility
from heca.scenes.scene import Scene
from heca.misc.state import State
from heca.misc.area import Area

from calvin_env_modified.envs.observation import CalvinEnvObservation
from tapas_gmm_modified.env.calvin import Calvin, CalvinConfig


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


class CalvinScene(Scene):
    @dataclass(kw_only=True)
    class Config(Scene.Config):
        label: str = "calvin"
        cam: str = "front"
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
        # self.scene_evaluator = SceneEvaluator(self)

        # v1 stuff
        self.overrides = set(
            [
                "block_red_position",
                "block_blue_position",
                "block_pink_position",
            ]
        )
        self.area_state = State(
            State.Config(
                values={"table", "drawer_open", "drawer_closed"},
                labeling=Area(CalvinAreaConfig()),
            ),
        )

    def close(self):
        self.env.close()

    def reset(self) -> tuple[DCScene, TDImage, np.ndarray]:
        obs = self.env.reset()[0]
        return self.from_internal(obs)

    def _step(self, action: np.ndarray) -> CalvinEnvObservation:
        return self.env.step(action, render=False)[0]

    def sample_task(
        self,
    ) -> tuple[
        tuple[DCScene, TDImage],
        tuple[DCScene, TDImage],
    ]:
        s_scene, s_images, _ = self.reset()
        g_scene, g_images, _ = self.reset()
        return (s_scene, s_images), (g_scene, g_images)

    def sample_task_vis(
        self,
    ) -> tuple[
        tuple[DCScene, TDImage, np.ndarray],
        tuple[DCScene, TDImage, np.ndarray],
    ]:
        return self.reset(), self.reset()

    @cached_property
    def entities(self) -> list[Entity]:
        ents = [
            Entity.Config(
                label="drawer",
                scene="calvin",
                question="Is the wooden, brown drawer under the wooden, brown table:",
                answers={"open", "closed"},
                states={"open", "closed"},
                mobility=Mobility.ARTICULATED,
            ),
            Entity.Config(
                label="slider",
                scene="calvin",
                question="Is the horizontal sliding door, with a grey handle, in the back of the table:",
                answers={"moved to the left", "moved to the right"},
                states={"left", "right"},
                mobility=Mobility.ARTICULATED,
            ),
            Entity.Config(
                label="button",
                scene="calvin",
                question="Is the black button:",
                answers={"pressed", "not pressed"},
                states={"pressed", "not pressed"},
                mobility=Mobility.STATIC,
            ),
            Entity.Config(
                label="switch",
                scene="calvin",
                question="Is the vertical switch:",
                answers={"up", "down"},
                states={"on", "off"},
                mobility=Mobility.ARTICULATED,
            ),
            Entity.Config(
                label="light",
                scene="calvin",
                question="Is the rectangular shaped light on the top left:",
                answers={"green ", "white"},
                states={"on", "off"},
                mobility=Mobility.STATIC,
            ),
            Entity.Config(
                label="red_block",
                scene="calvin",
                question="Is the red block:",
                answers={
                    "visible and in the brown, wooden drawer",
                    "visible and on the brown, wooden table",
                    "not visible anywhere",
                },
                states={"drawer", "table", "mia"},
                mobility=Mobility.FREE,
            ),
            Entity.Config(
                label="pink_block:",
                scene="calvin",
                question="Is the pink block:",
                answers={
                    "visible and in the brown, wooden drawer",
                    "visible and on the brown, wooden table",
                    "not visible anywhere",
                },
                states={"drawer", "table", "mia"},
                mobility=Mobility.FREE,
            ),
            Entity.Config(
                label="blue_block",
                scene="calvin",
                question="Is the blue block:",
                answers={
                    "visible and in the brown, wooden drawer",
                    "visible and on the brown, wooden table",
                    "not visible anywhere",
                },
                states={"drawer", "table", "mia"},
                mobility=Mobility.FREE,
            ),
        ]
        return [Entity.get(e) for e in ents]

    @cached_property
    def ee(self) -> Entity:
        config = Entity.Config(
            label="ee",
            scene="calvin",
            question="Is the robot arms gripper:",
            answers={"open", "closed"},
            states={"open", "closed"},
            mobility=Mobility.FREE,
        )
        return Entity.get(config)

    def get_gt_ee_values(
        self, obs: CalvinEnvObservation
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pos = torch.tensor(obs.ee_pose[:3], dtype=torch.float32)
        rot = torch.tensor(obs.ee_pose[-4:], dtype=torch.float32)
        state = torch.tensor([obs.ee_state], dtype=torch.float32)
        return pos, rot, state

    def get_dc_ee_values(
        self, obs: CalvinEnvObservation
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        soh = self.ee.state.one_hot_from_idx_dc(obs.ee_state.item())
        return obs.ee_pose[:3], obs.ee_pose[-4:], obs.ee_state, soh

    def is_tresholded(
        self, value: np.ndarray, target: float, threshold: float = 0.05
    ) -> bool:
        return bool((value < target - threshold) or (value > target + threshold))

    def gt_state(self, entity: Entity, state: np.ndarray) -> torch.Tensor:
        # Custom mehthod to parse states since calvin doesnt provide them as we need them.
        if entity.cfg.label == "drawer":
            if self.is_tresholded(state, 0.0):
                return entity.state.make_one_hot("closed")  # closed
            elif self.is_tresholded(state, 0.22):
                return entity.state.make_one_hot("open")  # open
            else:
                return entity.state.make_zeros()  # in between
        elif entity.cfg.label == "slider":
            if self.is_tresholded(state, 0.0):
                return entity.state.make_one_hot("left")  # left
            elif self.is_tresholded(state, 0.28):
                return entity.state.make_one_hot("right")  # right
            else:
                return entity.state.make_zeros()
        elif entity.cfg.label == "button":
            if self.is_tresholded(state, 0.0):
                return entity.state.make_one_hot("released")  # released
            elif self.is_tresholded(state, 1.0):
                return entity.state.make_one_hot("pressed")  # pressed
            else:
                return entity.state.make_zeros()  # in between
        elif entity.cfg.label == "light":
            if self.is_tresholded(state, 0.0):
                return entity.state.make_one_hot("off")  # off
            elif self.is_tresholded(state, 1.0):
                return entity.state.make_one_hot("on")  # on
            else:
                return entity.state.make_zeros()  # in between
        elif entity.cfg.label in {"red_block", "blue_block", "pink_block"}:
            return self.area_state(torch.Tensor(state))
        else:
            raise ValueError(f"Unknown entity label: {entity.cfg.label}")

    def dc_state(self, entity: Entity, state: np.ndarray) -> int:
        # Custom mehthod to parse states since calvin doesnt provide them as we need them.
        if entity.cfg.label == "drawer":
            if self.is_tresholded(state, 0.0):
                return entity.state.make_idx("closed")  # closed
            elif self.is_tresholded(state, 0.22):
                return entity.state.make_idx("open")  # open
            else:
                raise NotImplementedError
        elif entity.cfg.label == "slider":
            if self.is_tresholded(state, 0.0):
                return entity.state.make_idx("left")  # left
            elif self.is_tresholded(state, 0.28):
                return entity.state.make_idx("right")  # right
            else:
                raise NotImplementedError
        elif entity.cfg.label == "button":
            if self.is_tresholded(state, 0.0):
                return entity.state.make_idx("released")  # released
            elif self.is_tresholded(state, 1.0):
                return entity.state.make_idx("pressed")  # pressed
            else:
                raise NotImplementedError
        elif entity.cfg.label == "light":
            if self.is_tresholded(state, 0.0):
                return entity.state.make_idx("off")  # off
            elif self.is_tresholded(state, 1.0):
                return entity.state.make_idx("on")  # on
            else:
                raise NotImplementedError
        elif entity.cfg.label in {"red_block", "blue_block", "pink_block"}:
            raise NotImplementedError
        else:
            raise ValueError(f"Unknown entity label: {entity.cfg.label}")

    def to_dc_scene(self, obs: CalvinEnvObservation) -> DCScene:
        pos, rot, ste, soh = self.get_dc_ee_values(obs)
        td_entities: dict[str, DCEntity] = {}
        for entity in self.entities:
            e_pose = obs.object_poses.get(f"base__{entity.cfg.label}", None)
            e_ste = obs.object_states.get(f"base__{entity.cfg.label}", None)
            assert e_pose is not None, f"Missing pose for entity {entity.cfg.label}"
            assert e_ste is not None, f"Missing state for entity {entity.cfg.label}"
            e_pos = e_pose[:3]
            e_rot = e_pose[-4]
            e_ste = self.dc_state(entity, e_ste)
            e_soh = entity.state.one_hot_from_idx_dc(e_ste)
            td_entities[entity.cfg.label] = DCEntity(
                e_pos, e_rot, np.ndarray(e_ste), e_soh
            )
        ee = DCEntity(pos=pos, rot=rot, ste=ste, soh=soh)
        return DCScene(ee, td_entities)

    def to_np_image(self, obs: CalvinEnvObservation) -> np.ndarray:
        return obs.rgb["front"]

    def to_td_image(self, obs: CalvinEnvObservation) -> TDImage:
        rgb = obs.rgb["front"].transpose((2, 0, 1)) / 255
        mask = obs.mask["front"].astype(int)
        return TDImage(
            rgb=torch.Tensor(rgb),
            d=torch.Tensor(obs.depth["front"]),
            mask=torch.Tensor(mask).to(torch.uint8),
            extr=torch.Tensor(obs.extr["front"]),
            intr=torch.Tensor(obs.intr["front"]),
        )
