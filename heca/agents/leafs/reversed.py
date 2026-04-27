import re
import sys
import torch

from dataclasses import dataclass
from functools import cached_property
from tensordict import TensorDict

from heca.properties.v1 import PropertyV1
from heca.agents.leafs.tapas import TapasAgent
from heca.properties.default.v1.bool import BoolProperty
from heca.properties.default.v1.position import PositionProperty
from heca.properties.default.v1.rotation import RotationProperty
from heca.properties.evaluators.area import AreaEvaluator
from heca.properties.default.v1.area import AreaProperty, CalvinAreaConfig

import tapas_gmm_modified
from tapas_gmm_modified.utils.observation import SceneObservation

sys.modules["tapas_gmm"] = tapas_gmm_modified  # alias for unpickling old checkpoints

overrides: dict[str, list[PropertyV1.Config]] = {
    "close_drawer_back": [
        BoolProperty.Config(label="ee_scalar"),
        RotationProperty.Config(label="ee_rotation"),
        PositionProperty.Config(label="ee_position"),
    ],
    "open_drawer_back": [
        BoolProperty.Config(label="ee_scalar"),
        RotationProperty.Config(label="ee_rotation"),
        PositionProperty.Config(label="ee_position"),
    ],
    "press_button_back": [
        BoolProperty.Config(label="ee_scalar"),
        RotationProperty.Config(label="ee_rotation"),
        PositionProperty.Config(label="ee_position"),
    ],
    "open_slide_back": [
        BoolProperty.Config(label="ee_scalar"),
        RotationProperty.Config(label="ee_rotation"),
        PositionProperty.Config(label="ee_position"),
    ],
    "close_slide_back": [
        BoolProperty.Config(label="ee_scalar"),
        RotationProperty.Config(label="ee_rotation"),
        PositionProperty.Config(label="ee_position"),
    ],
    "place_red_table": [
        BoolProperty.Config(label="ee_scalar"),
        RotationProperty.Config(label="ee_rotation"),
        PositionProperty.Config(label="ee_position"),
        AreaProperty.Config(label="block_red_position"),
        RotationProperty.Config(label="block_red_rotation"),
        BoolProperty.Config(label="block_red_scalar"),
    ],
    "place_red_drawer": [
        BoolProperty.Config(label="ee_scalar"),
        RotationProperty.Config(label="ee_rotation"),
        PositionProperty.Config(label="ee_position"),
        AreaProperty.Config(label="block_red_position"),
        RotationProperty.Config(label="block_red_rotation"),
        BoolProperty.Config(label="block_red_scalar"),
    ],
    "place_pink_table": [
        BoolProperty.Config(label="ee_scalar"),
        RotationProperty.Config(label="ee_rotation"),
        PositionProperty.Config(label="ee_position"),
        AreaProperty.Config(label="block_pink_position"),
        RotationProperty.Config(label="block_pink_rotation"),
        BoolProperty.Config(label="block_pink_scalar"),
    ],
    "place_pink_drawer": [
        BoolProperty.Config(label="ee_scalar"),
        RotationProperty.Config(label="ee_rotation"),
        PositionProperty.Config(label="ee_position"),
        AreaProperty.Config(label="block_pink_position"),
        RotationProperty.Config(label="block_pink_rotation"),
        BoolProperty.Config(label="block_pink_scalar"),
    ],
    "place_blue_table": [
        BoolProperty.Config(label="ee_scalar"),
        RotationProperty.Config(label="ee_rotation"),
        PositionProperty.Config(label="ee_position"),
        AreaProperty.Config(label="block_blue_position"),
        RotationProperty.Config(label="block_blue_rotation"),
        BoolProperty.Config(label="block_blue_scalar"),
    ],
    "place_blue_drawer": [
        BoolProperty.Config(label="ee_scalar"),
        RotationProperty.Config(label="ee_rotation"),
        PositionProperty.Config(label="ee_position"),
        AreaProperty.Config(label="block_blue_position"),
        RotationProperty.Config(label="block_blue_rotation"),
        BoolProperty.Config(label="block_blue_scalar"),
    ],
}


class RTapasAgent(TapasAgent):
    @dataclass(kw_only=True)
    class Config(TapasAgent.Config):
        pos_reg: re.Pattern = re.compile(r"(.+?)_(?:position)")
        rot_reg: re.Pattern = re.compile(r"(.+?)_(?:rotation)")
        scalar_reg: re.Pattern = re.compile(r"(.+?)_(?:scalar)")

        def __post_init__(self):
            self.policy.invert_prediction_batch = True

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.evaluator = AreaEvaluator(AreaEvaluator.Config(area=CalvinAreaConfig()))
        self.overrides = overrides[self.cfg.query.label]

    def get_value(self, x: TensorDict, y: TensorDict) -> torch.Tensor:
        return self._hacky_reversed_fix(x, y)

    def _hacky_reversed_fix(self, x: SceneObservation, y: SceneObservation) -> SceneObservation:  # type: ignore
        for sig in self.overrides:
            pos = self.cfg.pos_reg.search(sig)
            rot = self.cfg.rot_reg.search(sig)
            scalar = self.cfg.scalar_reg.search(sig)
            if sig == "ee_position":
                x.ee_pose = torch.cat(
                    [
                        self.ppost[sig],
                        x.ee_pose[3:],
                    ]
                )
            elif sig == "ee_rotation":
                x.ee_pose = torch.cat(
                    [
                        x.ee_pose[:3],
                        self.ppost[sig],
                    ]
                )
            elif sig == "ee_scalar":
                x.ee_scalar = self.ppost[sig]
            elif pos:
                x.object_poses[pos.group(1)] = torch.cat(
                    [
                        self._hacky_position_fix(
                            y.object_poses[pos.group(1)][:3],
                            pos.group(1),
                        ),
                        x.object_poses[pos.group(1)][3:],
                    ]
                )
            elif rot:
                x.object_poses[rot.group(1)] = torch.cat(
                    [
                        x.object_poses[rot.group(1)][:3],
                        y.object_poses[rot.group(1)][3:],
                    ]
                )
            elif scalar:
                x.object_states[scalar.group(1)] = y.object_states[scalar.group(1)]
            else:
                raise ValueError(f"Unknown state name: {sig}")

        return x

    def _hacky_position_fix(self, x: torch.Tensor, label: str) -> torch.Tensor:
        if label in [  # TODO: check if names correct
            "block_red_position",
            "block_pink_position",
            "block_blue_position",
        ]:
            area = self.evaluator.area.check_eval_area(x)
            pos = x.clone()
            if (
                area == "drawer_closed"
            ):  # NOTE: This is a hardcoded fix for the drawer, since the position of the drawer in the reversed trajectory is in the closed position but the model expects it to be in the open position. This is because the model was trained with the open position as precondition and the closed position as postcondition, so when we reverse it, we need to also reverse the positions.
                pos[1] -= 0.17  # Drawer Offset
        else:
            pos = x
        return pos.numpy()

    @cached_property
    def demo_precons(self) -> dict[str, torch.Tensor]:
        return self.model.end_values

    @cached_property
    def demo_postcons(self) -> dict[str, torch.Tensor]:
        return self.model.start_values
