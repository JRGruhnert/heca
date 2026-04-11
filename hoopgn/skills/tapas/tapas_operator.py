from dataclasses import dataclass
from functools import cached_property
import re
from conf.hoopgnv1.properties.hoopgnv1 import PropertySet
from hoopgn.hardware import device
from loguru import logger
import numpy as np
import torch
from tapas_gmm_modified.policy import import_policy
from tapas_gmm_modified.policy.gmm import GMMPolicy, GMMPolicyConfig
from tapas_gmm_modified.policy.models.tpgmm import (
    AutoTPGMM,
    AutoTPGMMConfig,
    ModelType,
    TPGMMConfig,
)
from tapas_gmm_modified.utils.observation import SceneObservation
from tapas_gmm_modified.utils.robot_trajectory import (
    RobotTrajectory,
    TrajectoryPoint,
)
from hoopgn.objects.properties.property import Property
from hoopgn.observation.observation import StateValueDict
from hoopgn.skills.skill_operator import SkillOperator, SkillOperatorConfig
from hoopgn.objects.properties.features.evaluators.area_evaluator import (
    AreaEvaluator,
)
from hoopgn.objects.properties.features.conditions.condition import PropertyCondition


@dataclass(kw_only=True)
class TapasOperatorConfig(SkillOperatorConfig):
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


class TapasOperator(SkillOperator):
    def __init__(self, config: TapasOperatorConfig):
        super().__init__(config)
        self.config = config
        self.first_prediction = True
        self.predictions: RobotTrajectory | None = None
        self.prediction = None
        self.goal: StateValueDict | None = None
        self.pos_reg = re.compile(r"(.+?)_(?:position)")
        self.rot_reg = re.compile(r"(.+?)_(?:rotation)")
        self.scalar_reg = re.compile(r"(.+?)_(?:scalar)")
        self.hacky_properties = {
            "red_block_position": Property(config=PropertySet.block_red_position),
            "blue_block_position": Property(config=PropertySet.block_blue_position),
            "pink_block_position": Property(config=PropertySet.block_pink_position),
            "slide_position": Property(config=PropertySet.slide_position),
        }
        self.override = False
        if len(self.config.overrides):
            logger.warning(
                f"Skill {self.config.label} has overrides: {self.config.overrides}."
            )
            self.override = True

    def __call__(self, x: StateValueDict) -> np.ndarray | None:
        assert self.goal is not None, "Goal must be set before prediction."
        if self.config.predict_as_batch:
            if self.first_prediction:
                # NOTE: Could use control_duration later to enforce certain length
                try:
                    if self.override:
                        value = self._hacky_postprocess(x, self.goal)
                    else:
                        value = x
                    self.predictions, _ = self.policy.predict(value)  # type: ignore
                except FloatingPointError as e:
                    logger.error(f"Numerical error in GMM prediction: {e}")
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
                if self.override:
                    value = self._hacky_postprocess(x, self.goal)
                else:
                    value = x
                self.prediction, _ = self.policy.predict(value)  # type: ignore
            except FloatingPointError as e:
                logger.error(f"Numerical error in GMM prediction: {e}")
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

    def _hacky_postprocess(
        self,
        current: StateValueDict,
        goal: StateValueDict,
    ) -> SceneObservation:  # type: ignore
        for label, condition in self.overrides.items():
            pos = self.pos_reg.search(label)
            rot = self.rot_reg.search(label)
            scalar = self.scalar_reg.search(label)
            if label == "ee_position":
                current.tapas.ee_pose = torch.cat(
                    [
                        condition.value,
                        current.tapas.ee_pose[3:],
                    ]
                )
            elif label == "ee_rotation":
                current.tapas.ee_pose = torch.cat(
                    [
                        current.tapas.ee_pose[:3],
                        condition.value,
                    ]
                )
            elif label == "ee_scalar":
                current.tapas.ee_scalar = condition.value
            elif pos:
                current.tapas.object_poses[pos.group(1)] = np.concatenate(
                    [
                        self._hacky_position(pos.group(1), goal),
                        current.tapas.object_poses[pos.group(1)][3:],
                    ]
                )
            elif rot:
                current.tapas.object_poses[rot.group(1)] = np.concatenate(
                    [
                        current.tapas.object_poses[rot.group(1)][:3],
                        goal[f"{rot.group(1)}_rotation"].numpy(),
                    ]
                )
            elif scalar:
                current.tapas.object_states[scalar.group(1)] = goal[
                    f"{scalar.group(1)}_scalar"
                ].numpy()
            else:
                raise ValueError(f"Unknown state name: {label}")

        return current

    def _hacky_position(self, label: str, goal: StateValueDict) -> np.ndarray:
        property = self.hacky_properties[label]
        if isinstance(property.evaluator, AreaEvaluator):
            area = property.evaluator.area.check_eval_area(goal[label])
            temp_pos = goal[label].clone()
            if (
                area == "drawer_closed"
            ):  # NOTE: This is a hardcoded fix for the drawer, since the position of the drawer in the reversed trajectory is in the closed position but the model expects it to be in the open position. This is because the model was trained with the open position as precondition and the closed position as postcondition, so when we reverse it, we need to also reverse the positions.
                temp_pos[1] -= 0.17  # Drawer Offset
        else:
            temp_pos = goal[label]
        return temp_pos.numpy()

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
    def overrides(self) -> dict[str, PropertyCondition]:
        return self._load_overrides_conditions()

    def _load_overrides_conditions(self) -> dict[str, PropertyCondition]:
        return self._load_conditions(not self.config.reversed, self.config.overrides)

    def load_parameter_precons(self) -> dict[str, PropertyCondition]:
        return self._load_conditions(self.config.reversed)

    def load_parameter_postcons(self) -> dict[str, PropertyCondition]:
        return self._load_conditions(not self.config.reversed)

    def _load_conditions(
        self,
        reversed: bool,
        labels: set[str] | None = None,
    ) -> dict[str, PropertyCondition]:
        result = {}
        for l, c in self.config.conditions.items():
            if labels is None or l in labels:
                result[l] = PropertyCondition.from_demos(
                    (
                        self.demo_precons[l],
                        self.demo_postcons[l],
                        reversed,
                        l in self.tapas_labels,
                    ),
                    c,
                )
        return result

    def load_demo_precons(self) -> dict[str, torch.Tensor]:
        return self._load_demo_cons(self.config.reversed)

    def load_demo_postcons(self) -> dict[str, torch.Tensor]:
        return self._load_demo_cons(not self.config.reversed)

    def _load_demo_cons(self, reversed: bool) -> dict[str, torch.Tensor]:
        if reversed:
            return self.model.end_values
        else:
            return self.model.start_values

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
