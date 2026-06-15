from dataclasses import dataclass
from textwrap import dedent
from typing import Any, cast

import h5py
import numpy as np
import ogbench
import torch
from ogbench.manipspace.envs.scene_env import ManipSpaceEnv

from heca.entities.entity import Entity, Mobility
from heca.environment.scenes.scene import Scene
from heca.misc.td import (
    TDEntity,
    TDImage,
    TDScene,
    make_abs_and_rel_td_entity,
)


class TempScene(Scene):
    @dataclass(kw_only=True)
    class Config(Scene.Config):
        label: str = "ogbench"
        folder: str = "samples"
        id: str = "visual-scene-play-v0"
        cam: str = "front"
        mode: str = "task"  #
        ob_type: str = "pixels"  # states, pixels
        width: int = 256
        height: int = 256
        visualize_info: bool = False

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.reset_options = {"render_goal": True}

    def close(self):
        raise NotImplementedError()

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
    def entities(self) -> list[Entity]:
        ents = [
            Entity.Config(
                label="drawer_handle",
                question="What describes the drawer the best?",
                answers={"It is open", "It is closed"},
                states={"open", "closed"},  # 0, 1
                mobility=Mobility.ARTICULATED,
            ),
            Entity.Config(
                label="window_handle",
                question="What describes the sliding window the best?",
                answers={
                    "it is open and therefore moved to the front",
                    "it is closed and therefore moved to the back",
                },
                states={"open", "closed"},  # 0, 1
                mobility=Mobility.ARTICULATED,
            ),
            Entity.Config(
                label="button_0",
                question="What is the color of the left button?",
                answers={"white", "red"},
                states={"free", "locked"},  # 0, 1
                mobility=Mobility.STATIC,
            ),
            Entity.Config(
                label="button_1",
                question="What is the color of the right button?",
                answers={"white", "red"},
                states={"free", "locked"},  # 0, 1
                mobility=Mobility.STATIC,
            ),
            Entity.Config(
                label="block_0",
                question="Where is the red cube in the scene?",
                answers={
                    "inside the drawer",
                    "on the floor",
                    "unknown, cause it is not visible",
                },
                states={"drawer", "floor", "unknown"},  # 0, 1
                mobility=Mobility.FREE,
            ),
        ]
        return [Entity.get(e) for e in ents]

    @property
    def cursor(self) -> Entity:
        config = Entity.Config(
            label="cursor",
            question="Where is the red cube in the scene?",
            answers={
                "on the floor",
                "inside the drawer",
                "unknown, cause it is not visible",
            },
            states={"open", "closed"},  # 0, 1
            mobility=Mobility.FREE,
        )
        return Entity.get(config)

    def heca_td(self, obs: dict) -> TDScene:
        pos, rot, state = self.get_cursor(obs)
        td_entities: dict[str, TDEntity] = {}
        for entity in self.entities:
            e_pos = obs[f"privileged_{entity.cfg.label}_pos"]
            wxyz = obs[f"privileged_{entity.cfg.label}_quat"]
            e_rot = np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]], dtype=np.float32)
            e_state = obs[f"privileged_{entity.cfg.label}_state"]
            td_abs, td_rel = make_abs_and_rel_td_entity(
                position=torch.tensor(e_pos, dtype=torch.float32),
                rotation=torch.tensor(e_rot, dtype=torch.float32),
                state=entity.state.one_hot_from_idx(e_state),
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
        extras = self.get_extras(obs)
        return TDScene(td_entities, extras=extras)

    def to_td_scene_images(self, obs: dict) -> TDImage:
        image_dict = obs["image"]
        rgb = image_dict["rgb"].transpose((2, 0, 1)) / 255
        depth = image_dict["depth"]
        mask = image_dict["mask"]
        extr = image_dict["extrinsics"]
        intr = image_dict["intrinsics"]

        return TDImage(
            rgb=torch.Tensor(rgb),
            d=torch.Tensor(depth),
            mask=torch.Tensor(mask).to(torch.uint8),
            extr=torch.Tensor(extr),
            intr=torch.Tensor(intr),
        )

    def get_extras(self, obs: dict) -> dict[str, Any]:
        if "actions" in obs.keys():  # is demo
            action_raw = obs["actions"]
            yaw = action_raw[3]
            # quat = self.yaw_to_quat(yaw)
            axis_angle = np.array([0, 0, yaw])
            action = np.concatenate(
                [action_raw[:3], axis_angle, np.array([action_raw[4]])]
            )
            reward = obs["success"]
        else:
            pos = obs["proprio_effector_pos"]
            quat = obs["proprio_effector_quat"]
            state = obs["proprio_gripper_state"]
            action = np.concatenate([pos, quat, state])
            reward = np.array([0])
        return {
            "action": action,
            "reward": reward,
            "joint_pos": obs["proprio_joint_pos"],
            "joint_vel": obs["proprio_joint_vel"],
        }

    def to_internal(self, obs: Any, info: dict[str, Any]) -> Any:
        goal = info.pop("goal", None)
        goal_rendered = info.pop("goal_rendered", None)
        info["image"] = obs
        if goal is not None:
            goal["image"] = goal_rendered
        return info, goal

    def to_internal_goal(self, obs: Any, info: dict[str, Any]) -> Any:
        return obs

    def sample_task(
        self,
    ) -> tuple[
        tuple[TDScene, TDImage],
        tuple[TDScene, TDImage],
    ]:
        raise NotImplementedError()

    def sample_task_imaged(
        self,
    ) -> tuple[
        tuple[np.ndarray, TDImage],
        tuple[np.ndarray, TDImage],
    ]:
        raise NotImplementedError()

    def _step(self, action: np.ndarray) -> Any:
        action = self.to_internal_action(action)
        ob, reward, terminated, truncated, info = self.env.step(action)  # type: ignore
        obs, _ = self.to_internal(ob, info)
        return obs

    def get_cursor(self, obs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pos = torch.tensor(obs["proprio_effector_pos"], dtype=torch.float32)
        # wxyz = obs["proprio/effector_quat"]
        yaw = obs["proprio_effector_yaw"].item()
        rot = torch.tensor(self.yaw_to_quat(yaw), dtype=torch.float32)
        # rot = torch.tensor([wxyz[1], wxyz[2], wxyz[3], wxyz[0]], dtype=torch.float32)
        idx = obs["proprio_gripper_state"]
        state = self.cursor.state.one_hot_from_idx(idx)
        return pos, rot, state

    def yaw_to_quat(self, yaw: float) -> np.ndarray:
        half_yaw = yaw / 2
        return np.array([0, 0, np.sin(half_yaw), np.cos(half_yaw)], dtype=np.float32)

    def quat_to_yaw(self, quat: np.ndarray) -> float:
        # Assuming quat is in (x, y, z, w) format
        siny_cosp = 2 * (quat[3] * quat[2] + quat[0] * quat[1])
        cosy_cosp = 1 - 2 * (quat[1] ** 2 + quat[2] ** 2)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return yaw

    def to_internal_action(self, action: np.ndarray) -> np.ndarray:
        pos = action[:3]
        quat = action[3:7]
        yaw = self.quat_to_yaw(quat)
        gripper = action[7]
        return np.concatenate([pos, [yaw], [gripper]], axis=0)

    def load_dataset(
        self, file: h5py.File
    ) -> tuple[list[list[TDScene]], list[list[TDImage]]]:
        demo_indices: h5py.Dataset = file["demo"][:]  # type: ignore
        demo_length = len(demo_indices)
        pp_demos_scene: list[TDScene] = []
        pp_demos_image: list[TDImage] = []
        for i in range(demo_length):
            image = dict(
                rgb=file["rgb"][i],  # type: ignore
                depth=file["depth"][i],  # type: ignore
                mask=file["mask"][i],  # type: ignore
                extrinsics=file["extrinsics"][i],  # type: ignore
                intrinsics=file["intrinsics"][i],  # type: ignore
            )  # type: ignore
            ob: dict = {
                key: file[key][i]  # type: ignore
                for key in file.keys()
                if key
                not in [
                    "rgb",
                    "depth",
                    "mask",
                    "extrinsics",
                    "intrinsics",
                    "demo",
                ]
            }  # type: ignore
            # print(ob.keys())
            obs, _ = self.to_internal(image, ob)
            td_scene, td_image = self.from_internal(obs)
            pp_demos_scene.append(td_scene)
            pp_demos_image.append(td_image)

        change_points = np.where(np.diff(demo_indices) != 0)[0] + 1

        starts = np.concatenate([[0], change_points])
        ends = np.concatenate([change_points, [len(demo_indices)]])

        segments_scene: list[list[TDScene]] = []
        segments_image: list[list[TDImage]] = []
        for start, end in zip(starts, ends):
            segment_scene = pp_demos_scene[start:end]
            segment_image = pp_demos_image[start:end]
            segments_scene.append(segment_scene)
            segments_image.append(segment_image)
        return segments_scene, segments_image
