import sys
import torch
import tapas_gmm_modified
from dataclasses import dataclass
from functools import cached_property
from tapas_gmm_modified.utils.robot_trajectory import RobotTrajectory
from tapas_gmm_modified.policy.gmm import GMMPolicyConfig
from tapas_gmm_modified.policy.models.tpgmm import (
    AutoTPGMMConfig,
    ModelType,
    TPGMMConfig,
)

from heca.agents.experts.legacy import v1
from heca.agents.agent import Cursor
from heca.agents.experts.legacy.parameters.parameter import PropertyParameter
from heca.agents.experts.tapas import TapasAgent
from heca.environment.world import MetaWorld
from heca.misc.td import TDProperties

sys.modules["tapas_gmm"] = tapas_gmm_modified  # alias for unpickling old checkpoints


class LegacyTapas(TapasAgent):
    @dataclass(kw_only=True)
    class Config(TapasAgent.Config):
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
            invert_prediction_batch=False,
        )

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.prds: RobotTrajectory | None = None
        self.goal: TDProperties | None = None
        self.cursor = Cursor.IDLE
        self.parameters = {
            key: PropertyParameter.create(prop) for key, prop in v1.extractors.items()
        }

    def _load_conditions(self, pre: bool) -> dict[str, torch.Tensor]:
        if pre:
            precon = self.demo_precons
            postcon = self.demo_postcons
        else:
            precon = self.demo_postcons
            postcon = self.demo_precons

        result = {}
        for key, parameter in self.parameters.items():
            con = parameter.hoopgnv1(
                precon[key],
                postcon[key],
                key in self.tp_labels,
            )
            if con is not None:
                result[key] = con
        return result

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
    def ppre(self) -> dict[str, torch.Tensor]:
        return self._load_conditions(True)

    @cached_property
    def ppost(self) -> dict[str, torch.Tensor]:
        return self._load_conditions(False)

    @cached_property
    def demo_precons(self) -> dict[str, torch.Tensor]:
        return self.model.start_values

    @cached_property
    def demo_postcons(self) -> dict[str, torch.Tensor]:
        return self.model.end_values
