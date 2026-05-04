from dataclasses import dataclass

import numpy as np
from tensordict import TensorDict
import torch

from calvin_env_modified.envs.observation import (
    CalvinEnvObservation,
)
from tapas_gmm_modified.utils.observation import (
    CameraOrder,
    SingleCamObservation,
    SceneObservation,
    dict_to_tensordict,
)

from heca.environment.converters.converter import LeafConverter

empty_batchsize = torch.Size([])


class CalvinTapasConverter(LeafConverter):
    @dataclass(kw_only=True)
    class Config(LeafConverter.Config):
        label: str = "tapas"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def __call__(self, obs: CalvinEnvObservation) -> TensorDict:
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
