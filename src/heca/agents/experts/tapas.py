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
from heca.conditions.pair import ConPair
from heca.misc.data import DCScene, TDImage
from heca.misc.entity import Mobility
from heca.misc import logger
from heca.misc.hardware import device


class TapasAgent(ExpertAgent):
    @dataclass(kw_only=True)
    class Config(ExpertAgent.Config):
        label: str = "tapas"
        policy: GMMPolicyConfig = field(
            default_factory=lambda: GMMPolicyConfig(
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
            ),
        )
        repeat_actions: int = 0
        n_samples: int = 1000
        demo_filename: str = "demos_post.h5"
        gt_frames: list[list[int]] | None = None
        rel_score_threshold: float = 0.0
        demo_selections: list[int] | None = None

        def __post_init__(self):
            self.policy.model.frame_selection.gt_frames = self.gt_frames
            self.policy.model.frame_selection.rel_score_threshold = (
                self.rel_score_threshold
            )

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg

    def act(self, x: DCScene, y: DCScene) -> tuple[DCScene, AgentFeedback]:
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

    def make_scene(self, scene: DCScene, image: TDImage) -> DCScene:
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
        print(self.policy)

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
        assert isinstance(temp, AutoTPGMM)
        return temp

    @cached_property
    def demos(self) -> Demos:
        temp = self.model._demos
        assert isinstance(temp, Demos)
        return temp

    def tapas_td(self, dc_obs: DCScene, dc_goal: DCScene) -> TensorDict:
        action = torch.Tensor(dc_obs.extras["action"])
        reward = torch.Tensor(dc_obs.extras["reward"])
        joint_pos = torch.Tensor(dc_obs.extras["joint_pos"])
        joint_vel = torch.Tensor(dc_obs.extras["joint_vel"])
        ee_state = torch.Tensor(dc_obs.ee[-1])
        ee_pose = torch.cat(
            (torch.Tensor(dc_obs.ee[:3]), torch.Tensor(dc_obs.ee[3:7])),
            dim=-1,
        )

        # camera_obs = self.image_tensors(obs)
        # multicam_obs = dict_to_tensordict(
        #    {"_order": CameraOrder._create(obs.camera_names)} | camera_obs  # type: ignore
        # )
        poses = {
            entity.cfg.label: torch.cat(
                [
                    torch.Tensor(dc_obs[entity.cfg.label][:3]),
                    torch.Tensor(dc_obs[entity.cfg.label][3:7]),
                ],
                dim=-1,
            )
            for entity in sorted(self.scene.entities, key=lambda e: e.cfg.label)
        }

        states = {
            entity.cfg.label: torch.Tensor(dc_obs[entity.cfg.label][-1])
            for entity in sorted(self.scene.entities, key=lambda e: e.cfg.label)
        }

        for entity in sorted(self.scene.entities, key=lambda e: e.cfg.label):
            # This adds target frames for mobile entities.
            # Later can be used to set target for the tapas model
            if entity.cfg.mobility == Mobility.FREE:
                pos = torch.Tensor(dc_goal[entity.cfg.label][:3])
                rot = torch.Tensor(dc_goal[entity.cfg.label][3:7])
                state = torch.Tensor(dc_goal[entity.cfg.label][-1])
                pose = torch.cat((pos, rot), dim=-1)
                poses[f"{entity.cfg.label}_target"] = pose
                states[f"{entity.cfg.label}_target"] = state

        gee_state = torch.Tensor(dc_goal.ee[-1])
        gee_pose = torch.cat(
            (torch.Tensor(dc_goal.ee[:3]), torch.Tensor(dc_goal.ee[3:7])), dim=-1
        )

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
            batch_size=torch.Size([]),
        )

    @cached_property
    def elabels(self) -> set[str]:
        labels = set()
        elabels = [e.cfg.label for e in self.scene.entities]
        for idx, key in enumerate(self.demos.idx_key_list):
            if idx in self.model._used_frames and key in elabels:
                labels.add(key)
        return labels

    @cached_property
    def conditions(self) -> list[ConPair]:
        path = TapasAgent.load_dir(self.cfg)
        demos_file = h5py.File(path / self.cfg.demo_filename, "r")
        demos_scenes, demos_images = self.scene.load_dataset(
            demos_file,
            self.cfg.demo_selections,
            only_conditions=True,
        )

        pre_data: dict[str, np.ndarray] = {}
        post_data: dict[str, np.ndarray] = {}
        if self.cfg.use_gt:
            start_scenes = [demo[0] for demo in demos_scenes]
            end_scenes = [demo[-1] for demo in demos_scenes]
        else:
            start_scenes = [self.from_image(demo[0]) for demo in demos_images]
            end_scenes = [self.from_image(demo[-1]) for demo in demos_images]

        for key in self.elabels:
            pre_data[key] = np.stack([s[key] for s in start_scenes])
            post_data[key] = np.stack([s[key] for s in end_scenes])

        pre = Condition("pre", pre_data, 1, self.cfg.n_samples, self.cfg.threshold)
        post = Condition("post", post_data, 1, self.cfg.n_samples, self.cfg.threshold)
        pair = ConPair(f"{self.cfg.tag}_0", pre, post, self.cfg.threshold)
        pair.plot(path)
        return [pair]

    def load_demos(self, selections: list[int]) -> list[TensorDict]:
        path = TapasAgent.load_dir(self.cfg)
        demos_file = h5py.File(path / self.cfg.demo_filename, "r")

        observations: list[SceneObservation] = []  # type: ignore

        demos_scenes, demos_images = self.scene.load_dataset(
            demos_file, selections=selections
        )
        for i, (demo_scenes, demo_images) in enumerate(zip(demos_scenes, demos_images)):
            if self.cfg.use_gt:
                stacked = self.dcscenes_to_tdtapas(demo_scenes)
            else:
                demo_extracted: list[DCScene] = []
                for td_img in demo_images:
                    extracted = self.from_image(td_img)
                    demo_extracted.append(extracted)
                stacked = self.dcscenes_to_tdtapas(demo_extracted)
            observations.append(stacked)
        return observations

    def dcscenes_to_tdtapas(self, scenes: list[DCScene]) -> TensorDict:
        obs: list[TensorDict] = []
        td_goal = scenes[-1]
        for td_scene in scenes:
            td_obs = td_scene
            td = self.tapas_td(td_obs, td_goal)
            obs.append(td)
        stacked_obs = TensorDict.stack(obs, dim=0)
        assert isinstance(stacked_obs, SceneObservation)
        return stacked_obs  # type: ignore
