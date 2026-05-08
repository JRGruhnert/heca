import torch
import numpy as np
from dataclasses import dataclass

from tensordict import TensorDict

from heca.entities.entity import Entity
from heca.properties.property import Property
from heca.environment.scenes.scene import Scene
from heca.environment.scenes.calvin import v1, v2
from heca.environment.scenes.calvin.area import CalvinAreaConfig
from heca.misc.td import TDProperties, TDEntities, TDScene, empty_batchsize
from heca.misc.state import State

from calvin_env_modified.envs.observation import CalvinEnvObservation
from tapas_gmm_modified.env.calvin import Calvin, CalvinConfig
from tapas_gmm_modified.utils.observation import (
    CameraOrder,
    SingleCamObservation,
    SceneObservation,
    dict_to_tensordict,
)


class CalvinScene(Scene):
    @dataclass(frozen=True, kw_only=True)
    class Query(Scene.Query):
        label: str = "calvin"

    @dataclass(kw_only=True)
    class Config(Scene.Config):
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

        # v1 stuff
        self.overrides = set(
            [
                "block_red_position",
                "block_blue_position",
                "block_pink_position",
            ]
        )
        self.state = State.from_area_config(
            CalvinAreaConfig(),
        )
        self.valid = True

    def close(self):
        self.env.close()

    def _add_v1_td(self, td: TDScene, obs: CalvinEnvObservation) -> TDScene:
        td["v1"] = self.v1_td(obs)
        return td

    def _reset(self) -> CalvinEnvObservation:
        return self.env.reset()[0]

    def reset(self) -> TDScene:
        td = super().reset()
        td["v1"] = self.v1_td(self._reset())
        return td

    def _step(self, action: np.ndarray) -> CalvinEnvObservation:
        return self.env.step(action, render=False)[0]

    def step(self, action: np.ndarray) -> TDScene:
        td = super().step(action)
        td["v1"] = self.v1_td(self._step(action))
        return td

    def is_bad_sample(self, obs: TDScene) -> bool:
        return not self.valid

    def render(self):
        raise NotImplementedError()

    def tapas_td(
        self, obs: CalvinEnvObservation, extracted: TDEntities | None = None
    ) -> TensorDict:
        if obs.action is None:
            action = None
        else:
            action = torch.Tensor(obs.action)
        if obs.reward is None:
            reward = torch.Tensor([0.0])
        else:
            reward = torch.Tensor([obs.reward])
        joint_pos = torch.Tensor(obs.joint_pos)
        joint_vel = torch.Tensor(obs.joint_vel)
        ee_pose = torch.Tensor(obs.ee_pose)
        ee_state = torch.Tensor([obs.ee_state])
        camera_obs = self.image_tensors(obs)

        multicam_obs = dict_to_tensordict(
            {"_order": CameraOrder._create(obs.camera_names)} | camera_obs  # type: ignore
        )
        object_poses = dict_to_tensordict(
            {
                name: torch.Tensor(pose)
                for name, pose in sorted(obs.object_poses.items())
            },
        )
        object_states = dict_to_tensordict(
            {
                name: (
                    torch.tensor([state]) if np.isscalar(state) else torch.tensor(state)
                )
                for name, state in sorted(obs.object_states.items())
            },
        )

        return SceneObservation(
            feedback=reward,
            action=action,
            cameras=multicam_obs,
            ee_pose=ee_pose,
            gripper_state=ee_state,
            object_poses=object_poses,
            object_states=object_states,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            batch_size=empty_batchsize,
        )

    def image_tensors(self, obs: CalvinEnvObservation) -> dict[str, TensorDict]:
        camera_obs = {}
        for cam in obs.camera_names:
            rgb = obs.rgb[cam].transpose((2, 0, 1)) / 255
            mask = obs.mask[cam].astype(int)

            camera_obs[cam] = SingleCamObservation(
                **{
                    "rgb": torch.Tensor(rgb),
                    "d": torch.Tensor(obs.depth[cam]),
                    "mask": torch.Tensor(mask).to(torch.uint8),
                    "extr": torch.Tensor(obs.extr[cam]),
                    "intr": torch.Tensor(obs.intr[cam]),
                },
                batch_size=empty_batchsize,
            )
        return camera_obs

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
            state_dict[k] = torch.cat([state_dict[k], self.state(state_dict[k])])
            if torch.all(state_dict[k][3:] == 0):
                self.valid = False
        return TDProperties(state_dict)

    def properties(self) -> list[Property.Config]:
        return v1.properties

    def entities(self) -> list[Entity.Config]:
        return v2.entities
