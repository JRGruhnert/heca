from dataclasses import dataclass, field
from functools import cached_property
import re
import sys
from hoopgn.hardware import device
from hoopgn import logger
import numpy as np
import torch
import tapas_gmm_modified
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
from hoopgn.properties.property import Property, PropertyConfig
from hoopgn.observation.td_properties import TDProperties
from hoopgn.skills.skill_operator import SkillOperator, SkillOperatorConfig
from hoopgn.properties.features.evaluators.area_evaluator import (
    AreaEvaluator,
)
from hoopgn.properties.features.conditions.condition import PropertyCondition

sys.modules["tapas_gmm"] = tapas_gmm_modified  # alias for unpickling old checkpoints


@dataclass(kw_only=True)
class TapasOperatorConfig(SkillOperatorConfig):
    label: str
    reversed: bool
    overrides: set[str]
    pos_reg: re.Pattern = re.compile(r"(.+?)_(?:position)")
    rot_reg: re.Pattern = re.compile(r"(.+?)_(?:rotation)")
    scalar_reg: re.Pattern = re.compile(r"(.+?)_(?:scalar)")
    properties: list[PropertyConfig] = field(default_factory=list)  # type: ignore
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
        self.policy.invert_prediction_batch = self.reversed
        self.checkpoint_path = (
            "data/skills/tapas/" + self.label + "/demos_gmm_policy-release.pt"
        )


class TapasOperator(SkillOperator):
    def __init__(self, config: TapasOperatorConfig):
        super().__init__(config)
        self.config = config
        self.properties = {
            config.label: Property(config=config) for config in self.config.properties
        }
        self.new_task = False
        self.predictions: RobotTrajectory | None = None
        self.goal: TDProperties | None = None

    def __call__(self, x: TDProperties) -> np.ndarray | None:
        assert self.goal is not None, "Goal must be set before prediction."
        assert "tapas" in x.keys(), "Tapas observation must be present in the input."
        if self.new_task:
            try:
                if self.config.reversed:
                    value = self._hacky_postprocess(x["tapas"])
                else:
                    value = x["tapas"]
                self.predictions, _ = self.policy.predict(value)  # type: ignore
            except FloatingPointError as e:
                logger.error(f"Numerical error in prediction: {e}")
                return None
            except Exception as e:
                logger.error(f"General error in prediction: {e}")
                return None
            self.new_task = False
        if self.predictions is None or self.predictions.is_finished:
            return None
        return self._to_action(self.predictions.step())

    def reset(self, goal: TDProperties):
        self.policy.reset_episode()
        self.new_task = True
        self.predictions = None
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

    def _hacky_postprocess(self, tapas: SceneObservation) -> SceneObservation:  # type: ignore
        assert self.goal is not None, "Goal must be set for hacky postprocess."
        for label, condition in self.overrides.items():
            pos = self.config.pos_reg.search(label)
            rot = self.config.rot_reg.search(label)
            scalar = self.config.scalar_reg.search(label)
            if label == "ee_position":
                tapas.ee_pose = torch.cat(
                    [
                        condition.value,
                        tapas.ee_pose[3:],
                    ]
                )
            elif label == "ee_rotation":
                tapas.ee_pose = torch.cat(
                    [
                        tapas.ee_pose[:3],
                        condition.value,
                    ]
                )
            elif label == "ee_scalar":
                tapas.ee_scalar = condition.value
            elif pos:
                tapas.object_poses[pos.group(1)] = np.concatenate(
                    [
                        self._hacky_position(pos.group(1)),
                        tapas.object_poses[pos.group(1)][3:],
                    ]
                )
            elif rot:
                tapas.object_poses[rot.group(1)] = np.concatenate(
                    [
                        tapas.object_poses[rot.group(1)][:3],
                        self.goal[f"{rot.group(1)}_rotation"].numpy(),
                    ]
                )
            elif scalar:
                tapas.object_states[scalar.group(1)] = self.goal[
                    f"{scalar.group(1)}_scalar"
                ].numpy()
            else:
                raise ValueError(f"Unknown state name: {label}")

        return tapas

    def _hacky_position(self, label: str) -> np.ndarray:
        assert self.goal is not None, "Goal must be set for hacky position."
        evaluator = self.properties[label].evaluator
        if isinstance(evaluator, AreaEvaluator):
            area = evaluator.area.check_eval_area(self.goal[label])
            pos = self.goal[label].clone()
            if (
                area == "drawer_closed"
            ):  # NOTE: This is a hardcoded fix for the drawer, since the position of the drawer in the reversed trajectory is in the closed position but the model expects it to be in the open position. This is because the model was trained with the open position as precondition and the closed position as postcondition, so when we reverse it, we need to also reverse the positions.
                pos[1] -= 0.17  # Drawer Offset
        else:
            pos = self.goal[label]
        return pos.numpy()

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

    def load_precons(self) -> dict[str, PropertyCondition]:
        return self._load_conditions(self.config.reversed)

    def load_postcons(self) -> dict[str, PropertyCondition]:
        return self._load_conditions(not self.config.reversed)

    def _load_conditions(
        self,
        reversed: bool,
        labels: set[str] | None = None,
    ) -> dict[str, PropertyCondition]:
        logger.debug(
            f"Loading conditions with reversed={reversed} and labels={labels}."
        )
        result = {}
        logger.debug(
            f"Extracting Conditions from {len(self.config.properties)} properties."
        )
        for config in self.config.properties:
            result[config.label] = PropertyCondition.from_hoopgnv1_demos(
                value=(
                    self.demo_precons[config.label],
                    self.demo_postcons[config.label],
                    reversed,
                    config.label in self.tapas_labels,
                ),
                config=config,
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
        logger.info(f"Loading tapas operator from: {self.config.checkpoint_path}")
        temp = GMMPolicy(self.config.policy)
        assert isinstance(temp, GMMPolicy), "Policy model must be a GMMPolicy."
        temp.from_disk(self.config.checkpoint_path)
        temp.eval()
        return temp.to(device)

    @cached_property
    def model(self) -> AutoTPGMM:
        temp = self.policy.model
        assert isinstance(temp, AutoTPGMM), "Model must be an AutoTPGMM."
        return temp
