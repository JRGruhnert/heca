import pathlib
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import ogbench
import torch
from ogbench.manipspace.envs.scene_env import ManipSpaceEnv
from tensordict import TensorDict

from heca.entities.entity import Entity, Mobility
from heca.environment.scenes.scene import Scene
from heca.misc.td import (
    TDEntities,
    TDEntity,
    TDScene,
    TDSceneImages,
    make_abs_and_rel_td_entity,
)


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
        visualize_info: bool = True

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.reset_options = {"render_goal": True}
        self.env = cast(
            ManipSpaceEnv,
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

    @property
    def entities(self) -> list[Entity]:
        ents = [
            Entity.Config(
                label="drawer",
                states={"open", "open-locked", "closed", "closed-locked"},
                mobility=Mobility.ARTICULATED,
            ),
            Entity.Config(
                label="window",
                states={"open", "open-locked", "closed", "closed-locked"},
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
                label="red-block",
                states={"grabbed", "drawer", "floor", "mia"},
                mobility=Mobility.FREE,
            ),
        ]
        return [Entity.create(e) for e in ents]

    @property
    def cursor(self) -> Entity:
        config = Entity.Config(
            label="cursor",
            states={"open", "closed"},
            mobility=Mobility.FREE,
        )
        return Entity.create(config)

    def heca_td(self, obs: dict) -> TDEntities:
        pos, rot, state = self.get_cursor(obs)
        td_entities: dict[str, TDEntity] = {}
        for entity in self.entities:
            e_pos = obs[f""]
            e_rot = obs[f""]
            e_state = obs
        # e_pose = obs.object_poses.get(f"base__{entity.cfg.label}", None)
        # e_state = obs.object_states.get(f"base__{entity.cfg.label}", None)
        # assert e_pose is not None, f"Missing pose for entity {entity.cfg.label}"
        # assert e_state is not None, f"Missing state for entity {entity.cfg.label}"
        # final_kps = torch.tensor(e_pose, dtype=torch.float32)
        # final_state = self.gt_state(entity, e_state)
        # td_abs, td_rel = make_abs_and_rel_td_entity(
        #    position=final_kps[:3],
        #    rotation=final_kps[-4:],
        #    state=final_state,
        #    cursor_pos=pos,
        #    cursor_rot=rot,
        # )
        # td_entities[entity.cfg.label] = td_abs
        # td_entities[f"{entity.cfg.label}_rel"] = td_rel
        td_entities["cursor"] = TDEntity(
            position=pos,
            rotation=rot,
            state=state,
        )
        return TDEntities(td_entities)

    def to_td_scene_images(self, obs: dict) -> TDSceneImages:
        raise NotImplementedError()

    def sample_image(self) -> np.ndarray:
        raise NotImplementedError()

    def to_internal(self, obs: Any, info: dict[str, Any]) -> Any:
        goal = info["goal"]
        goal_rendered = info["goal_rendered"]
        info.pop("goal")
        info.pop("goal_rendered")
        info["image"] = obs
        goal["image"] = goal_rendered
        return info, goal

    def to_internal_goal(self, obs: Any, info: dict[str, Any]) -> Any:
        return obs

    def sample_task(
        self,
    ) -> tuple[
        tuple[TDScene, TDSceneImages],
        tuple[TDScene, TDSceneImages],
    ]:
        ob, info = self.env.reset(options=self.reset_options)
        obs, goal = self.to_internal(ob, info)
        s_scene, s_images = self.from_internal(obs)
        g_scene, g_images = self.from_internal(goal)
        return (s_scene, s_images), (g_scene, g_images)

    # [
    # 'proprio/joint_pos',
    # 'proprio/joint_vel',
    # 'proprio/effector_pos',
    # 'proprio/effector_yaw',
    # 'proprio/gripper_opening',
    # 'proprio/gripper_vel',
    # 'proprio/gripper_contact',
    # 'privileged/block_0_pos',
    # 'privileged/block_0_quat',
    # 'privileged/block_0_yaw',
    # 'privileged/button_0_state',
    # 'privileged/button_0_pos',
    # 'privileged/button_0_vel',
    # 'privileged/button_1_state',
    # 'privileged/button_1_pos',
    # 'privileged/button_1_vel',
    # 'privileged/drawer_pos',
    # 'privileged/drawer_vel',
    # 'privileged/drawer_handle_pos',
    # 'privileged/drawer_handle_yaw',
    # 'privileged/window_pos',
    # 'privileged/window_vel',
    # 'privileged/window_handle_pos',
    # 'privileged/window_handle_yaw',
    # 'prev_button_states',
    # 'button_states',
    # 'prev_qpos',
    # 'prev_qvel',
    # 'qpos',
    # 'qvel',
    # 'control',
    # 'time',
    # 'goal',
    # 'goal_rendered'
    # ]

    def _step(self, action: np.ndarray) -> Any:
        action = self.to_internal_action(action)
        ob, reward, terminated, truncated, info = self.env.step(action)  # type: ignore
        return self.to_internal(ob, info)

    def get_cursor(self, obs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def to_internal_action(self, action: np.ndarray) -> Any:
        return action  # TODO: from 8d to 5d

    # def to_tapas_format(self, obs) -> TensorDict:
    #    return td_scene  # TODO: convert to tapas format

    def _load(self, path: pathlib.Path):
        pass

    def test(self):
        (s_scene, s_images), (g_scene, g_images) = self.sample_task()
        print("Sampled task:")
        print("Initial scene:", s_scene)
        print("Goal scene:", g_scene)


# encoder Fix
# Dino Encoder Fix (padding)
# Encoder loading refs
# Image selector fix
