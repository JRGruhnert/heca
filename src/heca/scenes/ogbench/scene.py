from collections import defaultdict
from dataclasses import dataclass
from typing import Any, cast

import h5py
import numpy as np
import ogbench
import torch
from ogbench.manipspace.envs.scene_env import ManipSpaceEnv

from heca.data.data import DCEntity, DCScene, TDImage
from heca.scenes.scene import Scene


class OGScene(Scene):
    @dataclass(kw_only=True)
    class Config(Scene.Config):
        label: str = "ogbench"
        vis: bool
        tag: str

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.env_id = "vi-" + cfg.tag if cfg.vis else "gt-" + cfg.tag
        self.env = cast(
            ManipSpaceEnv,
            ogbench.make_env_and_datasets(
                dataset_name=self.env_id,
                env_only=True,
                dataset_only=False,
                control_timestep=0.5,
            ),
        )

    def close(self):
        self.env.close()

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
        pos = obs["proprio_effector_pos"]
        rot = obs["proprio_effector_quat"]
        ste = obs["proprio_gripper_opening"]
        rot = np.array([rot[1], rot[2], rot[3], rot[0]], dtype=np.float32)
        ee_pose = np.concatenate((pos, rot))
        if "actions" in obs.keys():  # is demo
            action_raw = obs["actions"]
            yaw = action_raw[3]
            axis_angle = np.array([0, 0, yaw])
            action = np.concatenate([action_raw[:3], axis_angle, ste])
            reward = obs["success"]
        else:
            yaw = obs["proprio_effector_yaw"].item()
            action = np.concatenate([pos, np.array([0, 0, yaw]), ste])
            reward = np.array([0])
        return {
            "action": action,
            "reward": reward,
            "ee_pose": ee_pose,
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
        ob, info = self.env.reset(options={"render_goal": True})
        obs, goal = self.to_internal(ob, info)
        self.last_pos = obs["proprio_effector_pos"]
        self.last_rot = obs["proprio_effector_yaw"]
        self.last_ste = obs["proprio_gripper_opening"]
        s_scene, s_image, _ = self.from_internal(obs)
        g_scene, g_image, _ = self.from_internal(goal)
        return (s_scene, s_image), (g_scene, g_image)

    def sample_task_vis(self) -> tuple[
        tuple[DCScene, TDImage, np.ndarray],
        tuple[DCScene, TDImage, np.ndarray],
    ]:
        ob, info = self.env.reset(options={"render_goal": True})
        obs, goal = self.to_internal(ob, info)
        self.last_pos = obs["proprio_effector_pos"]
        self.last_rot = obs["proprio_effector_yaw"]
        self.last_ste = obs["proprio_gripper_opening"]
        return self.from_internal(obs), self.from_internal(goal)

    def get_ee_dc(self, obs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pos = obs["proprio_effector_pos"]
        # wxyz = obs["proprio/effector_quat"]
        yaw = obs["proprio_effector_yaw"].item()
        rot = self.yaw_to_quat(yaw)
        # rot = torch.tensor([wxyz[1], wxyz[2], wxyz[3], wxyz[0]], dtype=torch.float32)
        ste_idx = np.atleast_1d(obs["proprio_gripper_state"])
        # print(
        #    f"ee {np.concatenate((self.last_pos, self.yaw_to_quat(yaw), self.last_state))}"
        # )
        return pos, rot, ste_idx

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
        result = np.concatenate([pos, [yaw], [state]], axis=0)
        return result

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

    def to_dc_scene(self, obs: dict) -> DCScene:
        dc_entities: dict[str, DCEntity] = {}
        for label, entity in self.entities.items():
            dc_entities[label] = entity.value_from_gt(label, obs)
        extras = self.get_extras(obs)
        return DCScene(dc_entities, extras=extras)

    def demo_auto_extract(self):
        scene_path = Scene.load_dir(self.cfg) / "demos"
        with h5py.File(scene_path / f"{self.env_id}.h5", "r") as f:
            done_ds = f["oracle_done"]
            assert isinstance(done_ds, h5py.Dataset)
            done = np.asarray(done_ds)[:, 0]
            success_ds = f["oracle_success"]
            assert isinstance(success_ds, h5py.Dataset)
            success = np.asarray(success_ds)[:, 0]
            task_ds = f["privileged_target_task"]
            assert isinstance(task_ds, h5py.Dataset)
            task = np.asarray(task_ds, dtype=str)

            # Find segment boundaries (indices where oracle switches)
            done_idxs = np.where(done == 1.0)[0]
            starts = [0] + list(done_idxs)
            ends = list(done_idxs) + [len(done)]

            agent_data = defaultdict(list)

            for s, e in zip(starts, ends):
                if s >= e:
                    continue
                t = e - 1
                if success[t] != 1.0:
                    continue

                label = str(task[t])
                agent_key = self.entities[label].make_agent_key(label, f, s, t)
                agent_data[agent_key].append((s, t))

            all_keys = list(f.keys())
            for agent_key, segments in agent_data.items():
                out_path = scene_path / f"{agent_key}.h5"
                with h5py.File(out_path, "w") as out:
                    demo_id = 0
                    for s, e in segments:
                        demo_ids = np.full(e - s + 1, demo_id, dtype=np.int32)
                        for key in all_keys:
                            ds = f[key]
                            if not isinstance(ds, h5py.Dataset):
                                continue
                            data = np.asarray(ds[s : e + 1])
                            if "demo" not in out and key == "demo":
                                continue  # skip if demo dataset doesn't exist yet
                            if key not in out:
                                maxshape = (None,) + data.shape[1:]
                                out.create_dataset(
                                    key,
                                    data=data,
                                    maxshape=maxshape,
                                    chunks=(1,) + data.shape[1:],
                                )
                            else:
                                ds = out[key]
                                assert isinstance(ds, h5py.Dataset)
                                n = ds.shape[0]
                                ds.resize(n + len(data), axis=0)
                                ds[n : n + len(data)] = data
                        # Add demo_id dataset
                        if "demo" not in out:
                            out.create_dataset(
                                "demo", data=demo_ids[:1], maxshape=(None,), chunks=(1,)
                            )
                        else:
                            ds = out["demo"]
                            assert isinstance(ds, h5py.Dataset)
                            n = ds.shape[0]
                            ds.resize(n + len(demo_ids), axis=0)
                            ds[n : n + len(demo_ids)] = demo_ids
                        demo_id += 1

                print(f"  {agent_key}: {len(segments)}, demo_path: {out_path}")
