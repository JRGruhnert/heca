from dataclasses import dataclass
from functools import cached_property
import sys
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
from tapas_gmm_modified.utils.robot_trajectory import (
    RobotTrajectory,
    TrajectoryPoint,
)
from hoopgn.observation.td_properties import TDProperties
from hoopgn.policies.leafs.leaf import LeafPolicy
from hoopgn.properties.property import Property
from hoopgn.misc.hardware import device
from hoopgn.misc import logger

sys.modules["tapas_gmm"] = tapas_gmm_modified  # alias for unpickling old checkpoints


class TapasPolicy(LeafPolicy):
    @dataclass(kw_only=True)
    class Config(LeafPolicy.Config):
        tapas: GMMPolicyConfig = GMMPolicyConfig(
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
            invert_prediction_batch=False,
        )

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.prds: RobotTrajectory | None = None
        self.goal: TDProperties | None = None
        self.properties: list[Property.Config] = []  # TODO: find a way to get that

    def __call__(self, x: TDProperties, y: TDProperties) -> np.ndarray | None:
        if self.prds is None or self.prds.is_finished:
            self.make_prediction(x, y)
        if self.prds is None:
            return None
        return self._to_action(self.prds.step())

    def get_value(self, x: TDProperties) -> torch.Tensor:
        return x["tapas"]

    def make_prediction(
        self, x: TDProperties, y: TDProperties
    ) -> RobotTrajectory | None:
        assert self.goal is not None, "Goal must be set before prediction."
        try:
            self.prds, _ = self.tapas.predict(self.get_value(x))  # type: ignore
        except FloatingPointError as e:
            logger.debug(f"Numerical error in prediction: {e}")
            return None
        except Exception as e:
            logger.debug(f"General error in prediction: {e}")
            return None

    def reset(self, goal: TDProperties):
        self.tapas.reset_episode()
        self.prds = None
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

    def _load_conditions(self, pre: bool) -> dict[Property.Signature, torch.Tensor]:
        if pre:
            precon = self.demo_precons
            postcon = self.demo_postcons
        else:
            precon = self.demo_postcons
            postcon = self.demo_precons

        result = {}
        for cfg in self.properties:
            con = Property.get(cfg.sig).extract_condition(
                precon[cfg.sig.label],
                postcon[cfg.sig.label],
                cfg in self.tp_labels,
            )
            if con is not None:
                result[cfg.sig] = con
        return result

    def from_disk(self, path: str):
        logger.info(f"Loading tapas policy from: {path}")
        temp = GMMPolicy(self.cfg.tapas)
        assert isinstance(temp, GMMPolicy), "Policy model must be a GMMPolicy."
        temp.from_disk(path)
        temp.eval()
        self.tapas = temp.to(device)

    @cached_property
    def model(self) -> AutoTPGMM:
        temp = self.tapas.model
        assert isinstance(temp, AutoTPGMM), "Model must be an AutoTPGMM."
        return temp

    @cached_property
    def tp_labels(self) -> set[str]:
        temp: set[str] = set()
        for _, segment in enumerate(self.model.segment_frames):  # type: ignore
            for _, frame_idx in enumerate(segment):
                pos_str, rot_str = self.model.frame_mapping[frame_idx]
                temp.add(pos_str)
                temp.add(rot_str)
        return temp

    @cached_property
    def ppre(self) -> dict[Property.Signature, torch.Tensor]:
        return self._load_conditions(True)

    @cached_property
    def ppost(self) -> dict[Property.Signature, torch.Tensor]:
        return self._load_conditions(False)

    @cached_property
    def demo_precons(self) -> dict[str, torch.Tensor]:
        return self.model.start_values

    @cached_property
    def demo_postcons(self) -> dict[str, torch.Tensor]:
        return self.model.end_values
