from dataclasses import dataclass
from textwrap import dedent
from typing import Any, cast

import h5py
import numpy as np
import ogbench
import torch
from ogbench.manipspace.envs.scene_env import ManipSpaceEnv

from heca.misc.data import DCEntity, DCScene, TDImage
from heca.misc.entity import Entity, Mobility
from heca.scenes.ogbench.utils import eval_pos, eval_state
from heca.scenes.scene import Scene


class OGBenchScene(Scene):
    @dataclass(kw_only=True)
    class Config(Scene.Config):
        label: str = "ogbench"
        tag: str = "samples"
        id: str = "visual-scene-play-v0"
        mode: str = "task"
        ob_type: str = "pixels"  # states, pixels
        width: int = 256
        height: int = 256
        visualize_info: bool = False

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
                # ob_type=cfg.ob_type,
                # mode=cfg.mode,
                # visualize_info=cfg.visualize_info,
                # width=cfg.width,
                # height=cfg.height,
                control_timestep=0.5,
            ),
        )

    def close(self):
        self.env.close()

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
                answers={"It is open", "It is closed"},
                states={"open", "closed"},  # 0, 1
                mobility=Mobility.ARTICULATED,
                eval_func=eval_pos,
            ),
            Entity.Config(
                label="window_handle",
                scene="ogbench",
                question="What describes the sliding window the best?",
                answers={
                    "it is open and therefore moved to the front",
                    "it is closed and therefore moved to the back",
                },
                states={"open", "closed"},  # 0, 1
                mobility=Mobility.ARTICULATED,
                eval_func=eval_pos,
            ),
            Entity.Config(
                label="button_0",
                scene="ogbench",
                question="What is the color of the left button?",
                answers={"white", "red"},
                states={"free", "locked"},  # 0, 1
                mobility=Mobility.STATIC,
                eval_func=eval_state,
            ),
            Entity.Config(
                label="button_1",
                scene="ogbench",
                question="What is the color of the right button?",
                answers={"white", "red"},
                states={"free", "locked"},  # 0, 1
                mobility=Mobility.STATIC,
                eval_func=eval_state,
            ),
            Entity.Config(
                label="block_0",
                scene="ogbench",
                question="Where is the red cube in the scene?",
                answers={
                    "inside the drawer",
                    "on the floor",
                    "unknown, cause it is not visible",
                },
                states={"drawer", "floor", "unknown"},  # 0, 1
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
            answers={
                "on the floor",
                "inside the drawer",
                "unknown, cause it is not visible",
            },
            states={"open", "closed"},  # 0, 1
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

    def to_td_image(self, obs: dict) -> TDImage:
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

    def to_np_image(self, obs: dict) -> np.ndarray:
        return obs["image"]["rgb"]

    def get_extras(self, obs: dict) -> dict[str, Any]:
        if "actions" in obs.keys():  # is demo
            action_raw = obs["actions"]
            # qpos = obs["prev_qpos"]
            # print(qpos)
            # assert False
            yaw = action_raw[3]
            # quat = self.yaw_to_quat(yaw)
            # = np.concatenate([action_raw[:3], quat, np.array([action_raw[4]])])
            axis_angle = np.array([0, 0, yaw])
            # gripper = action_raw[4]
            # action = np.concatenate([action_raw[:3], axis_angle, np.array([gripper])])
            opening = obs["proprio_gripper_opening"]
            action = np.concatenate([action_raw[:3], axis_angle, opening])
            reward = obs["success"]
        else:
            # print(obs.keys())
            # assert False
            pos = obs["proprio_effector_pos"]
            # quat = obs["proprio_effector_quat"]
            yaw = obs["proprio_effector_yaw"].item()
            state = obs["proprio_gripper_opening"]
            action = np.concatenate([pos, self.yaw_to_quat(yaw), state])
            print(f"bench {np.concatenate([pos, [yaw], state])}")
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

    def sample_task(
        self,
    ) -> tuple[
        tuple[DCScene, TDImage],
        tuple[DCScene, TDImage],
    ]:
        ob, info = self.env.reset(options=self.reset_options)
        obs, goal = self.to_internal(ob, info)
        self.last_pos = obs["proprio_effector_pos"]
        self.last_rot = obs["proprio_effector_yaw"]
        self.last_state = obs["proprio_gripper_opening"]
        s_scene, s_image, _ = self.from_internal(obs)
        g_scene, g_image, _ = self.from_internal(goal)
        return (s_scene, s_image), (g_scene, g_image)

    def sample_task_vis(self) -> tuple[
        tuple[DCScene, TDImage, np.ndarray],
        tuple[DCScene, TDImage, np.ndarray],
    ]:
        ob, info = self.env.reset(options=self.reset_options)
        obs, goal = self.to_internal(ob, info)
        self.last_pos = obs["proprio_effector_pos"]
        self.last_rot = obs["proprio_effector_yaw"]
        self.last_state = obs["proprio_gripper_opening"]
        return self.from_internal(obs), self.from_internal(goal)

    def get_ee(self, obs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pos = torch.tensor(obs["proprio_effector_pos"], dtype=torch.float32)
        # wxyz = obs["proprio/effector_quat"]
        yaw = obs["proprio_effector_yaw"].item()
        rot = torch.tensor(self.yaw_to_quat(yaw), dtype=torch.float32)
        # rot = torch.tensor([wxyz[1], wxyz[2], wxyz[3], wxyz[0]], dtype=torch.float32)
        idx = obs["proprio_gripper_state"]
        state = self.ee.one_hot_from_idx(idx)
        # print(
        #    f"ee {np.concatenate((self.last_pos, self.yaw_to_quat(yaw), self.last_state))}"
        # )
        return pos, rot, state

    def get_ee_dc(self, obs) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pos = obs["proprio_effector_pos"]
        # wxyz = obs["proprio/effector_quat"]
        yaw = obs["proprio_effector_yaw"].item()
        rot = self.yaw_to_quat(yaw)
        # rot = torch.tensor([wxyz[1], wxyz[2], wxyz[3], wxyz[0]], dtype=torch.float32)
        ste_idx = np.atleast_1d(obs["proprio_gripper_state"])
        ste_oh = self.ee.one_hot_from_idx_dc(ste_idx)
        # print(
        #    f"ee {np.concatenate((self.last_pos, self.yaw_to_quat(yaw), self.last_state))}"
        # )
        return pos, rot, ste_idx, ste_oh

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
        state = action[7]
        return np.concatenate([pos, [yaw], [state]], axis=0)

    def _step(self, action: np.ndarray) -> tuple[Any, float, bool, bool]:
        action = self.to_internal_action(action)
        ob, reward, terminated, truncated, info = self.env.unwrapped.step(action, False, True)  # type: ignore
        obs, _ = self.to_internal(ob, info)
        assert isinstance(reward, float)
        return obs, reward, terminated, truncated

    def load_dataset(
        self,
        file: h5py.File,
        selections: list[int] | None = None,
        only_conditions: bool = False,
    ) -> tuple[list[list[DCScene]], list[list[TDImage]]]:
        demo_indices: np.ndarray = file["demo"][:]  # type: ignore

        change_points = np.where(np.diff(demo_indices) != 0)[0] + 1
        starts = np.concatenate([[0], change_points])
        ends = np.concatenate([change_points, [len(demo_indices)]])

        segments_scene: list[list[DCScene]] = []
        segments_image: list[list[TDImage]] = []
        if selections is None:
            selections = list(range(len(starts)))

        for episode_idx in selections:
            start = starts[episode_idx]
            end = ends[episode_idx]

            segment_scene: list[DCScene] = []
            segment_image: list[TDImage] = []
            if only_conditions:
                indices = [start, end - 1]
            else:
                indices = range(start, end)
            for i in indices:
                image = dict(
                    rgb=file["rgb"][i],  # type: ignore
                    depth=file["depth"][i],  # type: ignore
                    mask=file["mask"][i],  # type: ignore
                    extrinsics=file["extrinsics"][i],  # type: ignore
                    intrinsics=file["intrinsics"][i],  # type: ignore
                )

                ob = {
                    key: file[key][i]  # type: ignore
                    for key in file.keys()
                    if key
                    not in {
                        "rgb",
                        "depth",
                        "mask",
                        "extrinsics",
                        "intrinsics",
                        "demo",
                    }
                }

                obs, _ = self.to_internal(image, ob)
                dc_scene, td_image, _ = self.from_internal(obs)

                segment_scene.append(dc_scene)
                segment_image.append(td_image)

            segments_scene.append(segment_scene)
            segments_image.append(segment_image)

        return segments_scene, segments_image
