from dataclasses import dataclass
from functools import cached_property
import re
from src.hardware import device
from src.observation.observation import empty_batchsize
from loguru import logger
import numpy as np
import torch

from calvin_env_modified.envs.observation import (
    CalvinEnvObservation,
)
from tapas_gmm.policy import import_policy
from tapas_gmm.policy.gmm import GMMPolicy, GMMPolicyConfig
from tapas_gmm.policy.models.tpgmm import (
    AutoTPGMM,
    AutoTPGMMConfig,
    ModelType,
    TPGMMConfig,
)
from tapas_gmm.utils.observation import (
    SceneObservation,
    SingleCamObservation,
    dict_to_tensordict,
)
from tapas_gmm.utils.robot_trajectory import (
    RobotTrajectory,
    TrajectoryPoint,
)
from src.observation.observation import StateValueDict
from src.skills.tree.operator import NodeOperator, NodeOperatorConfig
from src.states.logic.evaluations.evaluation_area import AreaEvalCondition
from src.states.logic.condition import Condition
from src.states.state import State


@dataclass
class TapasOperatorConfig(NodeOperatorConfig):
    label: str
    reversed: bool
    overrides: set[str]
    predict_as_batch: bool = True
    control_duration: int = -1
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
        binary_gripper_action=True,
        binary_gripper_closed_threshold=0.95,
        dbg_prediction=False,
        # the kinematics model in RLBench is just to unreliable -> leads to mistakes
        topp_in_t_models=False,
        force_overwrite_checkpoint_config=True,  # TODO:  otherwise it doesnt work
        time_scale=1.0,
        postprocess_prediction=False,  # TODO:  abs quaternions if False else delta quaternions
    )

    def __post_init__(self):
        self.policy.return_full_batch = self.predict_as_batch
        self.policy.batch_predict_in_t_models = self.predict_as_batch
        self.policy.invert_prediction_batch = self.reversed
        self.file_name = (
            "data/skills/tapas/" + self.label + "/demos_gmm_policy-release.pt"
        )


class TapasLeafOperator(NodeOperator):
    def __init__(self, config: TapasOperatorConfig):
        super().__init__(config)
        self.config = config
        self.first_prediction = True
        self.predictions: RobotTrajectory | None = None
        self.prediction = None
        self.goal: StateValueDict | None = None

    def __call__(
        self,
        current: CalvinEnvObservation,
        states: list[State],
    ) -> np.ndarray | None:
        assert self.goal is not None, "Goal must be set before prediction."
        assert isinstance(
            current, CalvinEnvObservation
        ), "Only supports CalvinEnvObservation."
        if self.config.predict_as_batch:
            if self.first_prediction:
                # NOTE: Could use control_duration later to enforce certain length
                try:
                    self.predictions, _ = self.policy.predict(  # type: ignore
                        self._to_skill_format(current, self.goal, states)
                    )
                except FloatingPointError as e:
                    logger.error(f"Numerical error in GMM prediction: {e}")
                    # Return a safe default action (e.g., no movement)
                    return None  # TODO: I think its just cause of a bad robot position
                except Exception as e:
                    logger.error(f"Error in skill prediction: {e}")
                    return None
                self.first_prediction = False
            if self.predictions is None or self.predictions.is_finished:
                return None
            return self._to_action(self.predictions.step())
        else:
            try:
                self.prediction, _ = self.policy.predict(  # type: ignore
                    self._to_skill_format(current, self.goal)
                )
            except FloatingPointError as e:
                logger.error(f"Numerical error in GMM prediction: {e}")
                # Return a safe default action (e.g., no movement)
                return None  # TODO: I think its just cause of a bad robot position
            except Exception as e:
                logger.error(f"Error in skill prediction: {e}")
                return None
            if self.prediction is None:
                return None
            return self._to_action(self.prediction)  # type: ignore

    def reset(self, goal: StateValueDict):
        self.policy.reset_episode()
        self.first_prediction = True
        self.predictions = None
        self.prediction = None
        self.goal = goal

    def _to_action(self, prediction: TrajectoryPoint | np.ndarray) -> np.ndarray:
        if isinstance(prediction, np.ndarray):
            return prediction
        return np.concatenate(
            (
                prediction.ee,
                prediction.gripper,
            )  # type: ignore
        )  # type: ignore

    def _to_skill_format(
        self,
        obs: CalvinEnvObservation,
        goal: StateValueDict | None = None,
        states: list[State] | None = None,
    ) -> SceneObservation:  # type: ignore
        """
        Convert the observation from the environment to a SceneObservation. This format is used for TAPAS.

        Returns
        -------
        SceneObservation
            The observation in common format as SceneObservation.
        """
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
                    "depth": torch.Tensor(obs.depth[cam]),
                    "mask": torch.Tensor(mask).to(torch.uint8),
                    "extr": torch.Tensor(obs.extr[cam]),
                    "intr": torch.Tensor(obs.intr[cam]),
                },
                batch_size=empty_batchsize,
            )

        multicam_obs = dict_to_tensordict(
            {"_order": CameraOrder._create(obs.camera_names)} | camera_obs  # type: ignore
        )
        object_poses_dict = obs.object_poses
        object_states_dict = obs.object_states
        if goal is not None and len(self.config.overrides) != 0:
            states_dict = {s.config.label: s for s in states} if states else {}
            # NOTE: This is only a hack to make reversed tapas models work
            # TODO: Update this when possible
            # logger.debug(f"Overriding Tapas Task {task.name}")
            for state_name, condition in self.overrides.items():
                state_value = (
                    condition.value.numpy() if condition.value is not None else None
                )
                assert state_value is not None, f"Condition {condition} has no value."
                match_position = re.search(r"(.+?)_(?:position)", state_name)
                match_rotation = re.search(r"(.+?)_(?:rotation)", state_name)
                match_scalar = re.search(r"(.+?)_(?:scalar)", state_name)
                if state_name == "ee_position":
                    ee_pose = torch.cat(
                        [
                            torch.Tensor(state_value),
                            ee_pose[3:],
                        ]
                    )
                elif state_name == "ee_rotation":
                    ee_pose = torch.cat(
                        [
                            ee_pose[:3],
                            torch.Tensor(state_value),
                        ]
                    )
                elif state_name == "ee_scalar":
                    ee_state = torch.Tensor(state_value)

                elif match_position:
                    # print(f"Overriding {state_name} in Tapas Skill {self.name}")
                    position_state_name = f"{match_position.group(1)}_position"
                    if position_state_name in states_dict:
                        temp_state = states_dict[position_state_name]
                        if isinstance(temp_state.eval_cnd, AreaEvalCondition):
                            area = temp_state.eval_cnd.area.check_eval_area(
                                goal[position_state_name]
                            )
                            temp_pos = goal[position_state_name].clone()
                            if (
                                area == "drawer_closed"
                            ):  # NOTE: This is a hardcoded fix for the drawer, since the position of the drawer in the reversed trajectory is in the closed position but the model expects it to be in the open position. This is because the model was trained with the open position as precondition and the closed position as postcondition, so when we reverse it, we need to also reverse the positions.
                                temp_pos[1] -= 0.17  # Drawer Offset
                        else:
                            temp_pos = goal[position_state_name]
                        object_poses_dict[match_position.group(1)] = np.concatenate(
                            [
                                temp_pos.numpy(),
                                object_poses_dict[match_position.group(1)][3:],
                            ]
                        )
                elif match_rotation:
                    object_poses_dict[match_rotation.group(1)] = np.concatenate(
                        [
                            object_poses_dict[match_rotation.group(1)][:3],
                            goal[f"{match_rotation.group(1)}_rotation"].numpy(),
                        ]
                    )
                elif match_scalar:
                    object_states_dict[match_scalar.group(1)] = goal[
                        f"{match_scalar.group(1)}_scalar"
                    ].numpy()
                else:
                    raise ValueError(f"Unknown state name: {state_name}")

        object_poses = dict_to_tensordict(
            {
                name: torch.Tensor(pose)
                for name, pose in sorted(object_poses_dict.items())
            },
        )
        object_states = dict_to_tensordict(
            {
                name: (
                    torch.tensor([state]) if np.isscalar(state) else torch.tensor(state)
                )
                for name, state in sorted(object_states_dict.items())
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

    def _load_demo_cons(self, reversed: bool) -> dict[str, torch.Tensor]:
        if reversed:
            return self.model.end_values
        else:
            return self.model.start_values

    @cached_property
    def demo_labels(self) -> set[str]:
        return set(self.demo_precons.keys()) | set(self.demo_postcons.keys())

    @cached_property
    def tapas_labels(self) -> set[str]:
        temp: set[str] = set()
        for _, segment in enumerate(self.model.segment_frames):  # type: ignore
            for _, frame_idx in enumerate(segment):
                pos_str, rot_str = self.model.frame_mapping[frame_idx]
                temp.add(pos_str)
                temp.add(rot_str)
        return temp

    @cached_property
    def overrides(self) -> dict[str, Condition]:
        return self._load_overrides_conditions()

    @cached_property
    def demo_precons(self) -> dict[str, torch.Tensor]:
        return self._load_demo_cons(self.config.reversed)

    @cached_property
    def demo_postcons(self) -> dict[str, torch.Tensor]:
        return self._load_demo_cons(not self.config.reversed)

    def load_parameter_precons(self) -> dict[str, Condition]:
        return self._load_conditions(self.config.reversed)

    def load_parameter_postcons(self) -> dict[str, Condition]:
        return self._load_conditions(not self.config.reversed)

    def _load_overrides_conditions(self) -> dict[str, Condition]:
        return self._load_conditions(not self.config.reversed, self.config.overrides)

    def _load_conditions(
        self,
        reversed: bool,
        labels: set[str] | None = None,
    ) -> dict[str, Condition]:
        result = {}
        for l, c in self.config.conditions.items():
            if labels is None or l in labels:
                cond = Condition.from_demos(
                    (
                        self.demo_precons[l],
                        self.demo_postcons[l],
                        reversed,
                        l in self.tapas_labels,
                    ),
                    c,
                )
                if cond is not None:
                    result[l] = cond
        return result

    @cached_property
    def policy(self) -> GMMPolicy:
        logger.info("Loading tapas operator from: {}", self.config.file_name)
        PolicyClass = import_policy("gmm")
        temp: GMMPolicy = PolicyClass(self.config.policy).to(device)
        temp.from_disk(self.config.file_name)
        temp.eval()
        return temp

    @cached_property
    def model(self) -> AutoTPGMM:
        temp = self.policy.model
        if not isinstance(temp, AutoTPGMM):
            raise ValueError(f"Expected AutoTPGMM, but got {type(temp)}.")
        return temp
