import sys
import numpy as np
import tapas_gmm_modified
from dataclasses import dataclass
from functools import cached_property
from tapas_gmm_modified.utils.robot_trajectory import RobotTrajectory
from tapas_gmm_modified.policy.gmm import GMMPolicy, GMMPolicyConfig
from tapas_gmm_modified.policy.models.tpgmm import AutoTPGMM
from tapas_gmm_modified.utils.observation import (
    CameraOrder,
    SingleCamObservation,
    SceneObservation,
    dict_to_tensordict,
)
from tensordict import TensorDict
import torch
from heca.agents.agent import AgentFeedback, Cursor
from heca.agents.experts.expert import ExpertAgent
from heca.entities.entity import Entity
from heca.misc.td import TDEntity, TDScene, empty_bs
from heca.misc import logger
from heca.misc.hardware import device

sys.modules["tapas_gmm"] = tapas_gmm_modified  # alias for unpickling old checkpoints


class TapasAgent(ExpertAgent):
    @dataclass(kw_only=True)
    class Config(ExpertAgent.Config):
        policy: GMMPolicyConfig

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg

    def act(self, x: TDScene, y: TDScene) -> tuple[TDScene, AgentFeedback]:
        self.policy.reset_episode()
        xt = self.tapas_td(x)
        yt = self.tapas_td(y)
        predictions = self.make_prediction(xt, yt)
        if predictions is None:
            return x, AgentFeedback(
                reward=0.0,
                done=True,
                terminal=True,
                state=Cursor.ERROR,
                learn=False,
            )
        self.state = Cursor.ACTIVE

        while not predictions.is_finished:
            pred = predictions.step()
            action = np.concatenate((pred.ee, pred.gripper))  # type: ignore
            td_scene, td_vision = self.scene.step(action)
        if self.cfg.use_gt:
            z = td_scene
        else:
            z = self.from_vision(td_vision)

        return z, self.evaluate(z, y)

    def evaluate(self, x: TDScene, y: TDScene) -> AgentFeedback:
        # TODO: implement evaluation logic
        return AgentFeedback(
            reward=0.0,
            done=True,
            terminal=True,
            state=self.state,
            learn=False,
        )

    def make_prediction(
        self, x: SceneObservation, y: SceneObservation  # type: ignore
    ) -> RobotTrajectory | None:
        try:
            prds, _ = self.policy.predict(x)  # type: ignore
            return prds  # type: ignore
        except FloatingPointError as e:
            logger.debug(f"Numerical error in prediction: {e}")
            return None
        except Exception as e:
            logger.debug(f"General error in prediction: {e}")
            return None

    def _load(self, path: str):
        logger.info(f"Loading tapas policy from: {path}")
        temp = GMMPolicy(self.cfg.policy)
        assert isinstance(temp, GMMPolicy), "Policy model must be a GMMPolicy."
        temp.from_disk(path)
        temp.eval()
        self.policy = temp.to(device)

    def _save(self, path: str):
        raise NotImplementedError("Saving not implemented for TapasAgent yet.")

    @cached_property
    def model(self) -> AutoTPGMM:
        temp = self.policy.model
        assert isinstance(temp, AutoTPGMM), "Model must be an AutoTPGMM."
        return temp

    def gen_pre(self) -> list[tuple[Entity, TDEntity]]:
        raise NotImplementedError()

    def gen_post(self) -> list[tuple[Entity, TDEntity]]:
        raise NotImplementedError()

    def tapas_td(self, td_scene: TDScene) -> TensorDict:
        action = None
        reward = torch.Tensor([0.0])
        # joint_pos = torch.Tensor(obs.joint_pos)
        # joint_vel = torch.Tensor(obs.joint_vel)
        cursor = td_scene["cursor"]
        cursor_pos = cursor.position
        cursor_rot = cursor.rotation
        cursor_state = cursor.state
        ee_pose = torch.cat((cursor_pos, cursor_rot), dim=-1)
        ee_state = cursor_state
        # camera_obs = self.image_tensors(obs)

        # multicam_obs = dict_to_tensordict(
        #    {"_order": CameraOrder._create(obs.camera_names)} | camera_obs  # type: ignore
        # )
        object_poses = dict_to_tensordict(
            {
                entity.cfg.label: torch.cat(
                    [
                        td_scene[entity.cfg.label].position,
                        td_scene[entity.cfg.label].rotation,
                    ],
                    dim=-1,
                )
                for entity in self.scene.entities
            },
        )
        object_states = dict_to_tensordict(
            {
                entity.cfg.label: (torch.tensor(td_scene[entity.cfg.label].state))
                for entity in self.scene.entities
            },
        )

        return SceneObservation(
            feedback=reward,
            action=action,
            cameras=None,  # multicam_obs,
            ee_pose=ee_pose,
            gripper_state=ee_state,
            object_poses=object_poses,
            object_states=object_states,
            joint_pos=None,  # joint_pos,
            joint_vel=None,  # joint_vel,
            batch_size=empty_bs,
        )

    # def image_tensors(self, obs: CalvinEnvObservation) -> dict[str, TensorDict]:
    #     camera_obs = {}
    #     for cam in obs.camera_names:
    #         rgb = obs.rgb[cam].transpose((2, 0, 1)) / 255
    #         mask = obs.mask[cam].astype(int)

    #         camera_obs[cam] = SingleCamObservation(
    #             **{
    #                 "rgb": torch.Tensor(rgb),
    #                 "d": torch.Tensor(obs.depth[cam]),
    #                 "mask": torch.Tensor(mask).to(torch.uint8),
    #                 "extr": torch.Tensor(obs.extr[cam]),
    #                 "intr": torch.Tensor(obs.intr[cam]),
    #             },
    #             batch_size=empty_bs,
    #         )
    #     return camera_obs
