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
from heca.experts.expert import ExpertModel
from heca.conditions.pair import ConPair
from heca.data.data import DCScene, TDImage
from heca.misc import logger
from heca.misc.hardware import device
from heca.scenes.scene import SceneFeedback


class TapasExpert(ExpertModel):
    @dataclass(kw_only=True)
    class Config(ExpertModel.Config):
        folder: str = "tapas"
        label: str = ""
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
                        distance_based=False,
                        velocity_based=True,
                        repeat_final_step=0,  # 1
                        components_prop_to_len=True,
                        velocity_threshold=0.05,
                    ),
                    cascade=CascadeConfig(),
                ),
                time_based=True,
                predict_dx_in_xdx_models=False,
                binary_gripper_action=False,
                binary_gripper_closed_threshold=0.0,
                dbg_prediction=True,
                force_overwrite_checkpoint_config=True,
                time_scale=1.0,
                postprocess_prediction=True,
                invert_prediction_batch=False,
                return_full_batch=True,
                batch_predict_in_t_models=True,
            ),
        )
        repeat_actions: int = 0
        gt_frames: list[list[int]] | None = None
        demo_selections: list[int] | None = None

        def __post_init__(self):
            self.policy.model.frame_selection.gt_frames = self.gt_frames

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg

    def _act(self, x: DCScene, y: DCScene) -> tuple[DCScene, SceneFeedback]:
        self.policy.reset_episode()
        xt = self.tapas_td(x, y)
        if self.cfg.policy.return_full_batch:
            predictions = self.make_batch_prediction(xt)
            if predictions is None:
                return x, SceneFeedback(
                    reward=0.0,
                    terminal=True,
                    truncated=False,
                )  # Error

            while not predictions.is_finished:
                pred = predictions.step()
                action = np.concatenate((pred.ee, pred.gripper))  # type: ignore
                # print(action.shape)
                tdscene, tdimage, fb = self.scene.step(action)
            z = self.make_scene(tdscene, tdimage)
        else:
            while not (pred := self.make_prediction(xt))[1]:
                action, _ = pred
                if action is None:
                    return x, SceneFeedback(
                        reward=0.0,
                        terminal=True,
                        truncated=False,
                    )  # Error
                tdscene, tdimage, fb = self.scene.step(action)
                z = self.make_scene(tdscene, tdimage)
                xt = self.tapas_td(z, y)

        return z, fb

    def make_scene(self, scene: DCScene, image: TDImage) -> DCScene:
        if self.cfg.use_gt:
            return scene
        else:
            return DCScene(self.from_image(image), scene.extras)

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
        filepath = path / "experts" / self.cfg.tag / file_name
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
        filepath = path / "experts" / self.cfg.tag / file_name
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
        poses = {l: dc_obs[l].tpose for l in self.scene.entities.keys()}

        for l in self.scene.entities.keys():
            poses[f"{l}_target"] = dc_goal[l].tpose
        poses["ee_target"] = torch.tensor(dc_goal.extras["ee_pose"])
        object_poses = dict_to_tensordict(poses)

        action = torch.Tensor(dc_obs.extras["action"])
        reward = torch.Tensor(dc_obs.extras["reward"])
        joint_pos = torch.Tensor(dc_obs.extras["joint_pos"])
        joint_vel = torch.Tensor(dc_obs.extras["joint_vel"])
        ee_pose = torch.tensor(dc_obs.extras["ee_pose"])

        return SceneObservation(
            feedback=reward,
            action=action,
            cameras=None,  # multicam_obs,
            ee_pose=ee_pose,
            # gripper_state=dc_obs.ee.tste,
            object_poses=object_poses,
            # object_states=object_states,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            batch_size=torch.Size([]),
        )

    def tps(self) -> set[str]:
        labels = set()
        for idx, key in enumerate(self.demos.idx_key_list):
            if idx in self.model._used_frames:
                assert key in self.scene.entities.keys()
                labels.add(key)
        logger.info(f"{self.cfg.tag} entities: {labels}")
        return labels

    @cached_property
    def conditions(self) -> ConPair:
        path = self.load_dir(self.cfg) / "demos"
        demos_file = h5py.File(path / f"{self.cfg.tag}.h5", "r")
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

        for key in self.entities:
            pre_data[key] = np.stack([s[key].value for s in start_scenes])
            post_data[key] = np.stack([s[key].value for s in end_scenes])

        pair = ConPair.make(self.cfg.tag, pre_data, post_data, self.entities, 1)
        pair.plot(path)
        return pair

    def fit_stage1(self, demos: Demos):
        self.model.fit_trajectories(
            demos,
            fix_frames=True,
            init_strategy=InitStrategy.TIME_BASED,
            fitting_actions=(FittingStage.INIT,),
        )

    def fit_stage2(self, demos: Demos):
        self.model.fit_trajectories(
            demos,
            fix_frames=True,
            fitting_actions=(FittingStage.EM_HMM,),
        )

    def plot_stage1(self):
        self.model.plot_model(
            scatter=True,
            annotate_gaussians=True,
            annotate_trajs=True,
            mean_as_base=False,
        )

    def plot_stage2(self):
        self.model.plot_model(
            scatter=True,
            annotate_gaussians=True,
            annotate_trajs=True,
            mean_as_base=False,
            time_based=True,
        )

    def load_demos(self, selections: list[int]) -> Demos:
        path = self.load_dir(self.cfg) / "demos"
        demos_file = h5py.File(path / f"{self.cfg.tag}.h5", "r")

        observations: list[SceneObservation] = []  # type: ignore

        demos_scenes, demos_images = self.scene.load_dataset(
            demos_file, selections=selections
        )
        for i, (demo_scenes, demo_images) in enumerate(zip(demos_scenes, demos_images)):
            if self.cfg.use_gt:
                stacked = self.dcscenes_to_tdtapas(demo_scenes)
            else:
                demo_extracted: list[DCScene] = []
                for idx, td_img in enumerate(demo_images):
                    extracted = self.from_image(td_img)
                    extr_scene = DCScene(extracted, demo_scenes[idx].extras)
                    demo_extracted.append(extr_scene)
                stacked = self.dcscenes_to_tdtapas(demo_extracted)
            observations.append(stacked)

        demos = Demos(
            observations,
            add_init_ee_pose_as_frame=True,
            add_world_frame=False,
            frames_from_keypoints=False,
            kp_indeces=None,
            enforce_z_up=False,
            modulo_object_z_rotation=False,
            make_quats_continuous=True,
        )  # type: ignore
        print("n_trajs", demos.n_trajs)
        print("n_frames", demos.n_frames)
        demos.frame_names
        return demos

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
