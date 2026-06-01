from functools import cached_property

import torch
import numpy as np
from dataclasses import dataclass, field
from heca.entities.entity import Entity, Mobility
from heca.image_extractors.image_extractor import ImageExtractor
from heca.environment.scenes.scene import Scene
from heca.misc.state import State
from heca.misc.area import Area
from heca.misc.td import (
    TDEntity,
    TDImage,
    TDProperties,
    TDEntities,
    TDScene,
    TDSceneVision,
    make_abs_and_rel_td_entity,
)

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
    @dataclass(frozen=True, kw_only=True)
    class Query(Scene.Query):
        label: str = "calvin"

    @dataclass(kw_only=True)
    class Config(Scene.Config):
        extractors: dict[str, ImageExtractor.Config] = field(
            default_factory=lambda: {
                # "wrist": ImageExtractor.Config(),
                "front": ImageExtractor.Config(),
            }
        )
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
        super().__init__(cfg)
        self.cfg = cfg
        self.env = Calvin(self.cfg.cc)

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

        self.valid = True

    def close(self):
        self.env.close()

    def _reset(self) -> CalvinEnvObservation:
        return self.env.reset()[0]

    def _step(self, action: np.ndarray) -> CalvinEnvObservation:
        return self.env.step(action, render=False)[0]

    def is_bad_sample(self, obs: TDScene) -> bool:
        return not self.valid

    def image_numpy(self, obs: CalvinEnvObservation) -> dict[str, np.ndarray]:
        return obs.rgb

    def to_td_scene_vision(self, obs: CalvinEnvObservation) -> TDSceneVision:
        camera_obs = {}
        for cam in obs.camera_names:
            rgb = obs.rgb[cam].transpose((2, 0, 1)) / 255
            mask = obs.mask[cam].astype(int)

            record = TDImage(
                rgb=torch.Tensor(rgb),
                d=torch.Tensor(obs.depth[cam]),
                mask=torch.Tensor(mask).to(torch.uint8),
                extr=torch.Tensor(obs.extr[cam]),
                intr=torch.Tensor(obs.intr[cam]),
            )
            camera_obs[cam] = record
        return TDSceneVision(camera_obs)

    def v1_td(self, obs: CalvinEnvObservation) -> TDProperties:
        state_dict = {}
        state_dict["ee_position"] = torch.tensor(obs.ee_pose[:3], dtype=torch.float32)
        state_dict["ee_rotation"] = torch.tensor(obs.ee_pose[-4:], dtype=torch.float32)
        state_dict["ee_scalar"] = torch.tensor(
            np.array([obs.ee_state]), dtype=torch.float32
        )

        for label, value in obs.object_poses.items():
            k = label.removeprefix("base__")
            state_dict[f"{k}_position"] = torch.tensor(value[:3], dtype=torch.float32)
            state_dict[f"{k}_rotation"] = torch.tensor(value[-4:], dtype=torch.float32)

        for label, value in obs.object_states.items():
            k = label.removeprefix("base__")
            state_dict[f"{k}_scalar"] = torch.tensor(
                np.array([value]), dtype=torch.float32
            )

        self.valid = True
        for k in self.overrides:
            state_dict[k] = torch.cat([state_dict[k], self.area_state(state_dict[k])])
            if torch.all(state_dict[k][3:] == 0):
                self.valid = False
        return TDProperties(state_dict)

    @cached_property
    def entities(self) -> list[Entity]:
        ents = [
            Entity.Config(
                label="drawer",
                states={"open", "closed"},
                mobility=Mobility.ARTICULATED,
            ),
            Entity.Config(
                label="slider",
                states={"left", "right", "middle"},
                mobility=Mobility.ARTICULATED,
            ),
            Entity.Config(
                label="button",
                states={"pressed", "released"},
                mobility=Mobility.STATIC,
            ),
            # Entity.Config(
            #    env="calvin",
            #    label="switch",
            #    states={"on", "off"},
            # ),
            Entity.Config(
                label="light",
                states={"on", "off"},
                mobility=Mobility.STATIC,
            ),
            Entity.Config(
                label="red_block",
                states={"grabbed", "drawer", "table", "mia"},
                mobility=Mobility.FREE,
            ),
            Entity.Config(
                label="pink_block",
                states={"grabbed", "drawer", "table", "mia"},
                mobility=Mobility.FREE,
            ),
            Entity.Config(
                label="blue_block",
                states={"grabbed", "drawer", "table", "mia"},
                mobility=Mobility.FREE,
            ),
        ]
        return [Entity.create(e) for e in ents]

    def get_cursor(
        self, obs: CalvinEnvObservation
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pos = torch.tensor(obs.ee_pose[:3], dtype=torch.float32)
        rot = torch.tensor(obs.ee_pose[-4:], dtype=torch.float32)
        state = torch.tensor([obs.ee_state], dtype=torch.float32)
        return pos, rot, state

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

    def heca_td(self, obs: CalvinEnvObservation) -> TDEntities:
        pos, rot, state = self.get_cursor(obs)
        td_entities: dict[str, TDEntity] = {}
        for entity in self.entities:
            e_pose = obs.object_poses.get(f"base__{entity.cfg.label}", None)
            e_state = obs.object_states.get(f"base__{entity.cfg.label}", None)
            assert e_pose is not None, f"Missing pose for entity {entity.cfg.label}"
            assert e_state is not None, f"Missing state for entity {entity.cfg.label}"
            final_kps = torch.tensor(e_pose, dtype=torch.float32)
            final_state = self.gt_state(entity, e_state)
            td_abs, td_rel = make_abs_and_rel_td_entity(
                position=final_kps[:3],
                rotation=final_kps[-4:],
                state=final_state,
                cursor_pos=pos,
                cursor_rot=rot,
            )
            td_entities[entity.cfg.label] = td_abs
            td_entities[f"{entity.cfg.label}_rel"] = td_rel
        td_entities["cursor"] = TDEntity(
            position=pos,
            rotation=rot,
            state=state,
        )
        return TDEntities(td_entities)
