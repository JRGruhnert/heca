from pathlib import Path
import h5py
import torch
import numpy as np
from dataclasses import dataclass, field
from functools import cached_property
from tensordict import TensorDict
from tapas_gmm_modified.utils.robot_trajectory import RobotTrajectory
from tapas_gmm_modified.policy.gmm import GMMPolicy, GMMPolicyConfig
from tapas_gmm_modified.utils.observation import SceneObservation, dict_to_tensordict
from tapas_gmm_modified.policy.models.tpgmm import (
    ModelType,
    FittingStage,
    InitStrategy,
    TPGMMConfig,
    AutoTPGMMConfig,
    AutoTPGMM,
    Demos,
    FrameSelectionConfig,
    DemoSegmentationConfig,
    CascadeConfig,
)
from heca.agents.agent import AgentFeedback
from heca.agents.experts.expert import ExpertAgent
from heca.conditions.condition import Condition
from heca.conditions.pair import ConditionPair
from heca.misc.entity import Mobility
from heca.misc.td import TDImage, TDScene, empty_bs
from heca.misc import logger
from heca.misc.hardware import device
from heca.scenes.scene import Scene


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
                    em_steps=50,
                    fix_first_component=False,  # True maybe
                    fix_last_component=False,  # True maybe
                    reg_init_diag=5e-4,  # 5
                    heal_time_variance=False,
                ),
                frame_selection=FrameSelectionConfig(
                    init_strategy=InitStrategy.TIME_BASED,
                    fitting_actions=(FittingStage.INIT,),
                    use_bic=False,
                    drop_redundant_frames=False,
                    rel_score_threshold=0.0,
                    gt_frames=None,  # Frames per segment
                ),
                demos_segmentation=DemoSegmentationConfig(
                    gripper_based=False,
                    distance_based=False,
                    velocity_based=True,
                    repeat_final_step=0,  # 1
                    repeat_first_step=0,
                    components_prop_to_len=True,
                    velocity_threshold=0.05,
                ),
                cascade=CascadeConfig(
                    kl_keep_time_dim=True,
                    kl_keep_rotation_dim=True,
                ),
            ),
            time_based=True,
            predict_dx_in_xdx_models=False,
            binary_gripper_action=False,
            binary_gripper_closed_threshold=0.0,
            dbg_prediction=False,
            force_overwrite_checkpoint_config=True,  # TODO:  otherwise it doesnt work
            time_scale=1.0,
            postprocess_prediction=True,  # TODO:  abs quaternions if False else delta quaternions
            invert_prediction_batch=False,
            return_full_batch=True,
            batch_predict_in_t_models=True,
        )
        repeat_actions: int = 0
        n_con_samples: int = 1000
        demo_filename: str = "demos_post.h5"
        gt_frames: list[list[int]] | None = None
        rel_score_threshold: float = 0.0

        def __post_init__(self):
            fs = self.policy.model.frame_selection
            fs.gt_frames = self.gt_frames
            fs.rel_score_threshold = self.rel_score_threshold

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
        temp = GMMPolicy(self.cfg.policy)
        assert isinstance(temp, GMMPolicy), "Policy model must be a GMMPolicy."
        if filepath.exists():
            temp.from_disk(str(filepath))
            logger.info(f"Loading tapas policy from: {filepath}")
        else:
            logger.warning(f"No tapas policy found at given path: {filepath}")
        self.policy = temp.to(device)

    def eval(self):
        self.policy.eval()

    def _save(self, path: Path):
        if self.cfg.use_gt:
            file_name = "policy_gt.pt"
        else:
            file_name = "policy_img.pt"
        filepath = path / file_name
        logger.info(f"Saving tapas policy to: {filepath}")
        self.model.to_disk(str(filepath))

    @cached_property
    def model(self) -> AutoTPGMM:
        temp = self.policy.model
        assert isinstance(temp, AutoTPGMM), "Model must be an AutoTPGMM."
        return temp

    def tapas_td(self, td_obs: TDScene, td_goal: TDScene) -> TensorDict:
        action = td_obs.extras["action"]
        reward = td_obs.extras["reward"]
        joint_pos = td_obs.extras["joint_pos"]
        joint_vel = td_obs.extras["joint_vel"]
        ee_state = td_obs["ee"].state
        ee_pose = torch.cat((td_obs["ee"].position, td_obs["ee"].rotation), dim=-1)

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

        gee_state = td_goal["ee"].state
        gee_pose = torch.cat((td_goal["ee"].position, td_goal["ee"].rotation), dim=-1)

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

    @cached_property
    def conditions(self) -> list[ConditionPair]:
        # https://distancia.readthedocs.io/en/latest/ChamferDistance.html
        tpgmm: AutoTPGMM = self.policy.model  # type: ignore
        assert tpgmm._demos is not None
        path = ExpertAgent.resolve(self.cfg)
        pre_data: dict[str, np.ndarray] = {}
        post_data: dict[str, np.ndarray] = {}
        for idx, key in enumerate(tpgmm._demos.idx_key_list):
            if idx in tpgmm._used_frames:
                pass
                # Use from h5 file
                # # precon
                # pre_pos = tpgmm._demos.start_pos[key].numpy()  # (N,3)
                # pre_rot = tpgmm._demos.start_rot[key].numpy()  # (N,4)
                # pre_state = tpgmm._demos.start_state[key].numpy()  # (N,C) or (N,)
                # if pre_state.ndim == 2:
                #     pre_state = np.argmax(pre_state, axis=1)

                # pre_data[key] = np.concatenate(
                #     (pre_pos, pre_rot, pre_state[:, None]), axis=1
                # )

                # # postcon
                # post_pos = tpgmm._demos.end_pos[key].numpy()
                # post_rot = tpgmm._demos.end_rot[key].numpy()
                # post_state = tpgmm._demos.end_state[key].numpy()  # (N,C) or (N,)
                # if post_state.ndim == 2:
                #     post_state = np.argmax(post_state, axis=1)

                # post_data[key] = np.concatenate(
                #     (post_pos, post_rot, post_state[:, None]), axis=1
                # )
        pre = Condition(
            f"{self.cfg.folder}_pre", pre_data, 1, self.cfg.n_con_samples, path
        )
        post = Condition(
            f"{self.cfg.folder}_post", post_data, 1, self.cfg.n_con_samples, path
        )
        return [ConditionPair(self.cfg.folder, pre, post)]

    def load_demos(
        self,
        add_init_ee_pose_as_frame=True,
        add_world_frame=False,
        frames_from_keypoints=False,
        kp_indeces=None,
        enforce_z_up=False,
        modulo_object_z_rotation=False,
        make_quats_continuous=True,
    ) -> Demos:
        path = TapasAgent.resolve(self.cfg)
        demos_file = h5py.File(path / self.cfg.demo_filename, "r")

        observations: list[SceneObservation] = []  # type: ignore

        if self.cfg.use_gt:
            load_scene = False
        else:
            load_scene = True
        scene = Scene.get(self.scene.cfg, load=load_scene)
        demos_scenes, demos_images = scene.load_dataset(demos_file)
        for i, (demo_scenes, demo_images) in enumerate(zip(demos_scenes, demos_images)):
            if self.cfg.use_gt:
                obss: list[TensorDict] = []
                for td_scene in demo_scenes:
                    td_obs = td_scene
                    td_goal = demo_scenes[-1]
                    obs = self.tapas_td(td_obs, td_goal)
                    obss.append(obs)
                stacked_obs = TensorDict.stack(obss, dim=0)
                observations.append(stacked_obs)
            else:
                raise NotImplementedError(
                    "TODO: implement tapas_td and convert demos to tapas format"
                )
                # TODO:

        return Demos(
            observations,
            add_init_ee_pose_as_frame=add_init_ee_pose_as_frame,
            add_world_frame=add_world_frame,
            frames_from_keypoints=frames_from_keypoints,
            kp_indeces=kp_indeces,
            enforce_z_up=enforce_z_up,
            modulo_object_z_rotation=modulo_object_z_rotation,
            make_quats_continuous=make_quats_continuous,
        )  # type: ignore
