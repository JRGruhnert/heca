from enum import Enum
import sys
from tensordict import TensorDict
import torch
import numpy as np
import tapas_gmm_modified
from dataclasses import dataclass
from functools import cached_property
from tapas_gmm_modified.utils.robot_trajectory import RobotTrajectory, TrajectoryPoint
from tapas_gmm_modified.policy.gmm import GMMPolicy, GMMPolicyConfig
from tapas_gmm_modified.policy.models.tpgmm import (
    AutoTPGMM,
    AutoTPGMMConfig,
    ModelType,
    TPGMMConfig,
)

from heca.agents.agent import AgentFeedback, Cursor
from heca.agents.scenes.scene_agent import SceneAgent
from heca.environments.world import MetaWorld
from heca.properties.property import Property
from heca.misc.td import TDScene, TDProperties, TDWorld
from heca.misc.hardware import device
from heca.misc import logger

sys.modules["tapas_gmm"] = tapas_gmm_modified  # alias for unpickling old checkpoints

ee_position = PositionProperty.Config(label="ee_position")
ee_rotation = RotationProperty.Config(label="ee_rotation")
ee_scalar = BoolProperty.Config(label="ee_scalar")
slide_position = PositionProperty.Config(label="slide_position")
slide_rotation = RotationProperty.Config(label="slide_rotation")
slide_scalar = RangeProperty.Config(label="slide_scalar", low=0.0, high=0.28)
drawer_position = PositionProperty.Config(label="drawer_position")
drawer_rotation = RotationProperty.Config(label="drawer_rotation")
drawer_scalar = RangeProperty.Config(label="drawer_scalar", low=0.0, high=0.22)
button_position = PositionProperty.Config(label="button_position")
button_rotation = RotationProperty.Config(label="button_rotation")
button_scalar = FlipProperty.Config(label="button_scalar")
led_position = PositionProperty.Config(label="led_position")
led_rotation = RotationProperty.Config(label="led_rotation")
block_red_position = AreaProperty.Config(label="block_red_position")
block_red_scalar = BoolProperty.Config(label="block_red_scalar")
block_blue_position = AreaProperty.Config(label="block_blue_position")
block_blue_scalar = BoolProperty.Config(label="block_blue_scalar")
block_pink_position = AreaProperty.Config(label="block_pink_position")
block_pink_scalar = BoolProperty.Config(label="block_pink_scalar")
block_red_rotation = RotationProperty.Config(
    label="block_red_rotation",
    evaluator=DefaultEvaluator.Config(),
)
block_blue_rotation = RotationProperty.Config(
    label="block_blue_rotation",
    evaluator=DefaultEvaluator.Config(),
)
block_pink_rotation = RotationProperty.Config(
    label="block_pink_rotation",
    evaluator=DefaultEvaluator.Config(),
)


class TapasAgent(SceneAgent):
    @dataclass(kw_only=True)
    class Config(SceneAgent.Config):
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
        self.scene = MetaWorld.get(self.cfg.scene)

    def predict(self, x: TDWorld, y: TDWorld) -> np.ndarray | None:
        if self.cursor == Cursor.IDLE:
            self.prds = self.make_prediction(x, y)
        if self.prds is None:
            self.cursor = Cursor.ERROR
            return None
        else:
            self.cursor = Cursor.RUNNING
            return self._to_action(self.prds.step())

    def act(self, x: TDScene, y: TDScene) -> tuple[TDScene, AgentFeedback]:
        z = x
        while (action := self.predict(z, y)) is not None:
            z = self.scene.step(action)

        return z, AgentFeedback(
            reward=0.0,
            done=True,
            terminal=True,
            cursor=self.cursor,
        )

    def get_value(self, x: TensorDict, y: TensorDict) -> torch.Tensor:
        return x

    def make_prediction(self, x: TDWorld, y: TDWorld) -> RobotTrajectory | None:
        tx = x.get(self.cfg.scene.label)["tapas"]
        ty = y.get(self.cfg.scene.label)["tapas"]
        try:
            prds, _ = self.policy.predict(self.get_value(tx, ty))  # type: ignore
            return prds  # type: ignore
        except FloatingPointError as e:
            logger.debug(f"Numerical error in prediction: {e}")
            return None
        except Exception as e:
            logger.debug(f"General error in prediction: {e}")
            return None

    def reset(self, goal: TDProperties):
        self.cursor = Cursor.IDLE
        self.policy.reset_episode()
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

    def _load_conditions(self, pre: bool) -> dict[Property.Config, torch.Tensor]:
        if pre:
            precon = self.demo_precons
            postcon = self.demo_postcons
        else:
            precon = self.demo_postcons
            postcon = self.demo_precons

        result = {}
        for key, props in self.entity.properties.items():
            prop = props[0]
            con = prop.extract_condition(
                precon[key],
                postcon[key],
                prop in self.tp_labels,
            )
            if con is not None:
                result[prop.cfg] = con
        return result

    def from_disk(self, path: str):
        logger.info(f"Loading tapas policy from: {path}")
        temp = GMMPolicy(self.cfg.policy)
        assert isinstance(temp, GMMPolicy), "Policy model must be a GMMPolicy."
        temp.from_disk(path)
        temp.eval()
        self.policy = temp.to(device)

    @cached_property
    def model(self) -> AutoTPGMM:
        temp = self.policy.model
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
    def ppre(self) -> dict[Property.Config, torch.Tensor]:
        return self._load_conditions(True)

    @cached_property
    def ppost(self) -> dict[Property.Config, torch.Tensor]:
        return self._load_conditions(False)

    @cached_property
    def postcon(self) -> dict[Property.Config, torch.Tensor]:
        raise NotImplementedError()

    @cached_property
    def precon(self) -> dict[Property.Config, torch.Tensor]:
        raise NotImplementedError()

    @cached_property
    def demo_precons(self) -> dict[str, torch.Tensor]:
        return self.model.start_values

    @cached_property
    def demo_postcons(self) -> dict[str, torch.Tensor]:
        return self.model.end_values
