from pathlib import Path
import torch
import numpy as np
from dataclasses import dataclass
from functools import cached_property
from tensordict import TensorDict
from tapas_gmm_modified.utils.robot_trajectory import RobotTrajectory
from tapas_gmm_modified.policy.gmm import GMMPolicy, GMMPolicyConfig
from tapas_gmm_modified.policy.models.tpgmm import AutoTPGMM
from tapas_gmm_modified.utils.observation import SceneObservation, dict_to_tensordict
from tapas_gmm_modified.utils.robot_trajectory import RobotTrajectory
from tapas_gmm_modified.policy.gmm import GMMPolicyConfig
from tapas_gmm_modified.policy.models.tpgmm import (
    AutoTPGMMConfig,
    ModelType,
    TPGMMConfig,
)

from heca.agents.agent import AgentFeedback
from heca.agents.experts.expert import ExpertAgent
from heca.entities.entity import Entity, Mobility
from heca.misc.td import TDEntity, TDImage, TDScene, empty_bs
from heca.misc import logger
from heca.misc.hardware import device


class TapasAgent(ExpertAgent):
    @dataclass(kw_only=True)
    class Config(ExpertAgent.Config):
        label: str = "tapas"
        policy: GMMPolicyConfig = GMMPolicyConfig(
            suffix="release",
            model=AutoTPGMMConfig(
                tpgmm=TPGMMConfig(
                    n_components=20,
                    model_type=ModelType.HMM,
                    use_riemann=True,
                    add_time_component=True,
                    add_action_component=False,
                    position_only=False,
                    add_gripper_action=True,
                    reg_shrink=1e-2,
                    reg_diag=2e-4,
                    reg_diag_gripper=2e-2,
                    reg_em_finish_shrink=1e-2,
                    reg_em_finish_diag=2e-4,
                    reg_em_finish_diag_gripper=2e-2,
                    trans_cov_mask_t_pos_corr=False,
                    em_steps=1,
                    fix_first_component=True,
                    fix_last_component=True,
                    reg_init_diag=5e-4,  # 5
                    heal_time_variance=False,
                ),
            ),
            time_based=True,
            predict_dx_in_xdx_models=False,
            binary_gripper_action=False,
            binary_gripper_closed_threshold=0.95,
            dbg_prediction=False,
            force_overwrite_checkpoint_config=True,  # TODO:  otherwise it doesnt work
            time_scale=1.0,
            postprocess_prediction=True,  # TODO:  abs quaternions if False else delta quaternions
            invert_prediction_batch=False,
            return_full_batch=True,
            batch_predict_in_t_models=True,
        )
        repeat_actions: int = 0

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg

    def act(self, x: TDScene, y: TDScene) -> tuple[TDScene, AgentFeedback]:
        self.policy.reset_episode()
        xt = self.tapas_td(x, y)
        if self.cfg.policy.return_full_batch:
            predictions = self.make_batch_prediction(xt)
            if predictions is None:
                return x, AgentFeedback(
                    reward=0,
                    done=False,
                    terminal=True,
                )  # Error

            while not predictions.is_finished:
                pred = predictions.step()
                action = np.concatenate((pred.ee, pred.gripper))  # type: ignore
                tdscene, tdimage, reward, done, truncated = self.scene.step(action)
            z = self.make_scene(tdscene, tdimage)
        else:
            while not (pred := self.make_prediction(xt))[1]:
                action, _ = pred
                if action is None:
                    return x, AgentFeedback(
                        reward=0,
                        done=False,
                        terminal=True,
                    )  # Error
                tdscene, tdimage, reward, done, truncated = self.scene.step(action)
                z = self.make_scene(tdscene, tdimage)
                xt = self.tapas_td(z, y)

        return z, AgentFeedback(
            reward=reward,
            done=done,
            terminal=truncated,
        )

    def make_scene(self, scene: TDScene, image: TDImage):
        if self.cfg.use_gt:
            return scene
        else:
            return self.from_image(image)

    def make_batch_prediction(
        self, x: SceneObservation  # type: ignore
    ) -> RobotTrajectory | None:
        # prds, _ = self.policy.predict(x)
        try:
            prds, _ = self.policy.predict(x)  # type: ignore
            return prds  # type: ignore
        except Exception as e:
            logger.debug(f"Error: {e}")
            return None

    def make_prediction(self, x: SceneObservation) -> tuple[np.ndarray | None, bool]:  # type: ignore
        # prds, _ = self.policy.predict(x)
        try:
            prds, info = self.policy.predict(x)  # type: ignore
            return prds, info["done"]  # type: ignore
        except Exception as e:
            logger.debug(f"Error: {e}")
            return None, True

    def _load(self, path: Path):
        # logger.info()
        if self.cfg.use_gt:
            file_name = "policy_gt.pt"
        else:
            file_name = "policy_img.pt"
        filepath = path / file_name
        logger.info(f"Loading tapas policy from: {filepath}")
        temp = GMMPolicy(self.cfg.policy)
        assert isinstance(temp, GMMPolicy), "Policy model must be a GMMPolicy."
        temp.from_disk(str(filepath))
        temp.eval()
        self.policy = temp.to(device)

    def _save(self, path: Path):
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

    def tapas_td(self, td_obs: TDScene, td_goal: TDScene) -> TensorDict:
        action = td_obs.extras["action"]
        reward = td_obs.extras["reward"]
        joint_pos = td_obs.extras["joint_pos"]
        joint_vel = td_obs.extras["joint_vel"]
        cursor = td_obs["cursor"]
        cursor_pos = cursor.position
        cursor_rot = cursor.rotation
        cursor_state = cursor.state
        ee_pose = torch.cat((cursor_pos, cursor_rot), dim=-1)
        ee_state = cursor_state

        # camera_obs = self.image_tensors(obs)
        # multicam_obs = dict_to_tensordict(
        #    {"_order": CameraOrder._create(obs.camera_names)} | camera_obs  # type: ignore
        # )
        poses = {
            entity.cfg.label: torch.cat(
                [
                    td_obs[entity.cfg.label].position,
                    td_obs[entity.cfg.label].rotation,
                ],
                dim=-1,
            )
            for entity in self.scene.entities
        }

        states = {
            entity.cfg.label: td_obs[entity.cfg.label].state
            for entity in self.scene.entities
        }

        for entity in self.scene.entities:
            # This adds target frames for mobile entities.
            # Later can be used to set target for the tapas model
            if entity.cfg.mobility == Mobility.FREE:
                pos = td_goal[entity.cfg.label].position
                rot = td_goal[entity.cfg.label].rotation
                state = td_goal[entity.cfg.label].state
                pose = torch.cat((pos, rot), dim=-1)
                poses[f"{entity.cfg.label}_target"] = pose
                states[f"{entity.cfg.label}_target"] = state

        gcursor = td_goal["cursor"]
        gcursor_pos = gcursor.position
        gcursor_rot = gcursor.rotation
        gcursor_state = gcursor.state
        gee_pose = torch.cat((gcursor_pos, gcursor_rot), dim=-1)
        gee_state = gcursor_state
        poses[f"ee_target"] = gee_pose
        states[f"ee_target"] = gee_state

        object_poses = dict_to_tensordict(poses)
        object_states = dict_to_tensordict(states)

        return SceneObservation(
            feedback=reward,
            action=action,
            cameras=None,  # multicam_obs,
            ee_pose=ee_pose,
            gripper_state=ee_state,
            object_poses=object_poses,
            object_states=object_states,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            batch_size=empty_bs,
        )
