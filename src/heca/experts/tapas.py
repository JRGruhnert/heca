from pathlib import Path
import h5py
import joblib
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
from heca.misc.interrupt import stop_requested
from heca.misc.hardware import device
from heca.scenes.scene import Scene, SceneFeedback

# --- riepybdlib quaternion-log singularity patch ---------------------------------
# The S³ quaternion manifold's log map divides by the norm of the quaternion's
# vector part, so it is singular at the antipode [-1, 0, 0, 0] (a full 360°
# rotation). ogbench's end-effector points down via a ~180° roll about the x-axis
# (w ≈ 0), and TAPAS's per-frame projection can wrap the relative orientation
# through 360°, producing w ≈ -1 and NaN covariances during fitting. Projecting
# each quaternion onto the positive-real (w ≥ 0) hemisphere before the log makes
# the map single-valued and avoids the singularity without changing the data
# convention. This lives here (rather than in the pinned riepybdlib dependency)
# because we can only change this code base.
import riepybdlib.mappings as _rbd_mappings

_orig_quat_log_e = _rbd_mappings.quat_log_e


def _hemisphere_quat_log_e(g, reg=1e-6, arccos_func=_rbd_mappings.arccos_cont):
    if isinstance(g, list):
        g = [(-q if q.q0 < 0 else q) for q in g]
    elif g.q0 < 0:
        g = -g
    return _orig_quat_log_e(g, reg, arccos_func)


def _hemisphere_quat_log_e_star(g, reg=1e-6):
    return _hemisphere_quat_log_e(g, reg, _rbd_mappings.arccos_star)


_rbd_mappings.quat_log_e = _hemisphere_quat_log_e
_rbd_mappings.quat_log_e_star = _hemisphere_quat_log_e_star
# --------------------------------------------------------------------------------


class TapasExpert(ExpertModel):
    @dataclass(kw_only=True)
    class Config(ExpertModel.Config):
        folder: str = "tapas"
        policy: GMMPolicyConfig = field(
            default_factory=lambda: GMMPolicyConfig(
                suffix="release",
                model=AutoTPGMMConfig(
                    tpgmm=TPGMMConfig(
                        n_components=10,
                        model_type=ModelType.HMM,
                        use_riemann=True,
                        add_time_component=True,
                        add_action_component=False,
                        position_only=False,
                        add_gripper_action=True,
                        reg_shrink=1e-3,  # 1e-2
                        reg_diag=1e-3,  # 2e-4
                        reg_diag_gripper=2e-2,
                        reg_em_finish_shrink=1e-2,
                        reg_em_finish_diag=2e-4,
                        reg_em_finish_diag_gripper=2e-2,
                        trans_cov_mask_t_pos_corr=False,
                        em_steps=50,
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
                        components_prop_to_len=True,  # True,
                        velocity_threshold=0.006,
                    ),
                    cascade=CascadeConfig(),
                ),
                time_based=True,
                predict_dx_in_xdx_models=True,
                binary_gripper_action=False,
                binary_gripper_closed_threshold=0.0,
                dbg_prediction=False,
                force_overwrite_checkpoint_config=True,
                time_scale=1.0,
                postprocess_prediction=True,
                invert_prediction_batch=False,
                return_full_batch=True,
                batch_predict_in_t_models=True,
            ),
        )
        repeat_actions: int = 0
        gt_frames: list[list[str]] | None = None
        demo_selections: list[int] | None = None
        segment_ids: list[int] | None = None
        max_con_comps: int = 10

        def __post_init__(self):
            # Convert entity-name-based gt_frames to the frame indices the
            # underlying TPGMM expects. frame_names = ["ee_init"] + entity
            # labels + f"{label}_target" labels + ["ee_target"], matching the
            # object_poses keys built by tapas_td.
            if self.gt_frames is not None:
                flat = [n for seg in self.gt_frames for n in seg]
                if flat and all(isinstance(n, str) for n in flat):
                    scene = Scene.get(self.scene, auto_load=False)
                    labels = list(scene.entities.keys())
                    frame_names = (
                        ["ee_init"]
                        + labels
                        + [f"{l}_target" for l in labels]
                        + ["ee_target"]
                    )
                    name_to_idx = {name: i for i, name in enumerate(frame_names)}
                    gt_frames = [
                        [name_to_idx[n] for n in seg] for seg in self.gt_frames
                    ]
                    self.policy.model.frame_selection.gt_frames = gt_frames

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        temp = GMMPolicy(self.cfg.policy)
        assert isinstance(temp, GMMPolicy), "Policy model must be a GMMPolicy."
        self.policy = temp.to(device)

    def _act(self, x: DCScene, y: DCScene) -> tuple[DCScene, SceneFeedback]:
        self.policy.reset_episode()
        xt = self.tapas_td(x, y)
        if self.cfg.policy.return_full_batch:
            predictions = self.make_batch_prediction(xt)
            if predictions is None:
                return x, SceneFeedback(reward=0.0, terminal=True, truncated=False)
            while not predictions.is_finished:
                if stop_requested():
                    # Abort mid-rollout so the worker thread returns promptly
                    # during shutdown instead of finishing the whole trajectory.
                    return x, SceneFeedback(reward=0.0, terminal=True, truncated=True)
                pred = predictions.step()
                action = np.concatenate((pred.ee, pred.gripper))  # type: ignore
                # print(action)
                tdscene, tdimage, fb = self.scene.step(action)
            z = self.make_scene(tdscene, tdimage)
        else:
            while not (pred := self.make_prediction(xt))[1]:
                if stop_requested():
                    return x, SceneFeedback(reward=0.0, terminal=True, truncated=True)
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

    def _act_virt(self, x: DCScene, y: DCScene) -> tuple[DCScene, SceneFeedback]:
        tdscene, tdimage, fb = self.scene.step_virt(x, y, list(self.entities.keys()))
        return self.make_scene(tdscene, tdimage), fb

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
        filepath = self.policy_path(path)
        if filepath.exists():
            self.policy.from_disk(str(filepath))
            self.policy = self.policy.to(device)
            logger.info(f"Loading tapas policy from: {filepath}")
        else:
            logger.warning(f"No tapas policy found at given path: {filepath}")

    def eval(self):
        self.policy.eval()

    def policy_path(self, path: Path) -> Path:
        if self._use_gt:
            file_name = "policy_gt.pt"
        else:
            file_name = "policy_img.pt"
        return path / file_name

    def _save(self, path: Path):
        filepath = self.policy_path(path)
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
        poses = {l: dc_obs[l].tpose for l in self.scene.entities}

        for l in self.scene.entities:
            poses[f"{l}_target"] = dc_goal[l].tpose
        # Goal ee pose, in the same normalized frame as the object targets
        # (previously center_yaw_pose, a constant workspace-center position).
        poses["ee_target"] = torch.tensor(dc_goal.extras["ee_pose"])
        object_poses = dict_to_tensordict(poses)

        states = {l: dc_obs[l].tste for l in self.scene.entities}

        for l in self.scene.entities:
            states[f"{l}_target"] = dc_goal[l].tste
        states["ee_target"] = torch.tensor(dc_goal.extras["gripper_state"])
        object_states = dict_to_tensordict(states)

        action = torch.Tensor(dc_obs.extras["action"])
        reward = torch.Tensor(dc_obs.extras["reward"])
        joint_pos = torch.Tensor(dc_obs.extras["joint_pos"])
        joint_vel = torch.Tensor(dc_obs.extras["joint_vel"])
        ee_pose = torch.tensor(dc_obs.extras["ee_pose"])
        gripper_state = torch.Tensor(dc_obs.extras["gripper_state"])

        return SceneObservation(
            feedback=reward,
            action=action,
            cameras=None,  # multicam_obs,
            ee_pose=ee_pose,
            gripper_state=gripper_state,
            object_poses=object_poses,
            object_states=object_states,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            batch_size=torch.Size([]),
        )

    def tps(self) -> set[str]:
        labels = set()
        for idx, key in enumerate(self.demos.frame_names):
            if idx in self.model._used_frames and key in self.scene.entities:
                labels.add(key)
        logger.info(f"{self.cfg.tag} entities: {labels}")
        return labels

    @cached_property
    def conditions(self) -> ConPair:
        path = self.load_dir(self.cfg)
        cache_path = path / "conditions.joblib"
        if cache_path.exists() and not self._force_recompute:
            logger.info(f"Loading cached conditions from {cache_path}")
            return joblib.load(cache_path)

        demos_file = h5py.File(path / f"demos.h5", "r")
        demos_scenes, demos_images = self.scene.load_dataset(
            demos_file,
            self.cfg.demo_selections,
            only_conditions=True,
            with_images=not self._use_gt,
        )

        pre_data: dict[str, np.ndarray] = {}
        post_data: dict[str, np.ndarray] = {}
        if self._use_gt:
            start_scenes = [demo[0] for demo in demos_scenes]
            end_scenes = [demo[-1] for demo in demos_scenes]
        else:
            start_scenes = [self.from_image(demo[0]) for demo in demos_images]
            end_scenes = [self.from_image(demo[-1]) for demo in demos_images]

        for key in self.entities:
            pre_data[key] = np.stack([s[key].value for s in start_scenes])
            post_data[key] = np.stack([s[key].value for s in end_scenes])

        pair = ConPair.make(self.cfg.tag, pre_data, post_data, self.entities)
        pair.plot(path)
        joblib.dump(pair, cache_path)
        logger.info(f"Saved conditions to {cache_path}")
        return pair

    def fit_stage1(self, demos: Demos):
        liks, avg_logliks = self.model.fit_trajectories(
            demos,
            init_strategy=InitStrategy.TIME_BASED,
            fitting_actions=(FittingStage.INIT,),
        )
        logger.info(f"stage1 avg_logliks={avg_logliks}")
        return liks, avg_logliks

    def fit_stage2(self, demos: Demos):
        liks, avg_logliks = self.model.fit_trajectories(
            demos,
            fitting_actions=(FittingStage.EM_HMM,),
        )
        logger.info(f"stage2 avg_logliks={avg_logliks}")
        return liks, avg_logliks

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

    def plot_demo_velocities(
        self, selections: list[int] | None = None, max_demos: int | None = None
    ):
        """Plot velocity + segmentation for each demo into ``plots/vel/``.

        By default plots every demo id present in this agent's ``demos.h5``.
        Pass ``selections`` to restrict to specific demo ids, or ``max_demos``
        to only plot the first ``max_demos`` ids.
        """
        import matplotlib.pyplot as plt

        if selections is None:
            selections = self._all_demo_ids()
        if max_demos is not None:
            selections = selections[:max_demos]

        demos = self.load_demos(selections)
        trajs = demos.get_action_magnitude(subsampled=False, position_only=False)
        if isinstance(trajs, torch.Tensor):
            trajs = (trajs,)

        seg_cfg = self.cfg.policy.model.demos_segmentation
        out_dir = self.save_dir(self.cfg) / "plots" / "vel"
        out_dir.mkdir(parents=True, exist_ok=True)

        for demo_id, traj in zip(selections, trajs):
            vel = np.asarray(traj[..., 0].detach().cpu())
            boundaries = self._velocity_segments(vel, seg_cfg)

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(np.arange(len(vel)), vel, linewidth=0.5, c="gray")
            ax.axhline(y=seg_cfg.velocity_threshold, color="r", label="threshold")
            for m in boundaries:
                ax.axvline(x=m, color="g", linestyle="--", label="segment")
            ax.set_title(f"{self.cfg.tag} — demo {demo_id}")
            ax.set_xlabel("timestep")
            ax.set_ylabel("velocity (position magnitude)")

            path = out_dir / f"demo_{demo_id}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Saved demo velocity plot to {path}")

    def _all_demo_ids(self) -> list[int]:
        path = self.load_dir(self.cfg) / "demos.h5"
        with h5py.File(path, "r") as f:
            demo = np.asarray(f["demo"][:])  # type: ignore
        if len(demo) == 0:
            return []
        return list(range(int(np.max(demo)) + 1))

    @staticmethod
    def _velocity_segments(
        vel: np.ndarray, seg_cfg: DemoSegmentationConfig
    ) -> list[int]:
        stop_indeces = np.argwhere(np.abs(vel) < seg_cfg.velocity_threshold).flatten()
        if len(stop_indeces) == 0:
            return []
        idx_diff = np.diff(stop_indeces)
        split_idx = np.argwhere(idx_diff > seg_cfg.max_idx_distance).flatten() + 1
        if len(split_idx) == 0:
            stop_segmented = [stop_indeces]
        else:
            stop_segmented = np.split(stop_indeces, split_idx)
        filtered = [c for c in stop_segmented if len(c) >= seg_cfg.min_len]
        segment_mean = [int(c.mean()) for c in filtered]
        return [
            m
            for m in segment_mean
            if m > seg_cfg.min_end_distance and m < len(vel) - seg_cfg.min_end_distance
        ]

    def load_demos(self, selections: list[int] | None = None) -> Demos:
        if selections is None:
            selections = self.cfg.segment_ids or []
        path = self.load_dir(self.cfg)
        demos_file = h5py.File(path / f"demos.h5", "r")

        observations: list[SceneObservation] = []  # type: ignore

        demos_scenes, demos_images = self.scene.load_dataset(
            demos_file, selections=selections, with_images=not self._use_gt
        )
        for i, demo_scenes in enumerate(demos_scenes):
            if self._use_gt:
                stacked = self.dcscenes_to_tdtapas(demo_scenes)
            else:
                demo_extracted: list[DCScene] = []
                for idx, td_img in enumerate(demos_images[i]):
                    extracted = self.from_image(td_img)
                    extr_scene = DCScene(extracted, demo_scenes[idx].extras)
                    demo_extracted.append(extr_scene)
                stacked = self.dcscenes_to_tdtapas(demo_extracted)
            observations.append(stacked)

        return Demos(
            observations,
            meta_data={"tag": self.cfg.tag + self.cfg.label},
            add_init_ee_pose_as_frame=True,
            add_world_frame=False,
            frames_from_keypoints=False,
            kp_indeces=None,
            enforce_z_up=False,
            modulo_object_z_rotation=False,
            make_quats_continuous=True,
        )  # type: ignore

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
