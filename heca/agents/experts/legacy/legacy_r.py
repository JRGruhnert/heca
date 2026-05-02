import re
import sys
import torch

from dataclasses import dataclass
from functools import cached_property

from heca.agents.experts.legacy.lagacy_n import LegacyTapas
from heca.properties.default.v1.area import CalvinAreaConfig
from heca.misc.area import Area

import tapas_gmm_modified
from tapas_gmm_modified.utils.observation import SceneObservation

sys.modules["tapas_gmm"] = tapas_gmm_modified  # alias for unpickling old checkpoints

common_overrides = [
    "ee_scalar",
    "ee_rotation",
    "ee_position",
]
red_overrides = common_overrides + [
    "block_red_scalar",
    "block_red_rotation",
    "block_red_position",
]
pink_overrides = common_overrides + [
    "block_pink_scalar",
    "block_pink_rotation",
    "block_pink_position",
]
blue_overrides = common_overrides + [
    "block_blue_scalar",
    "block_blue_rotation",
    "block_blue_position",
]
overrides: dict[str, list[str]] = {
    "close_drawer_back": common_overrides,
    "open_drawer_back": common_overrides,
    "press_button_back": common_overrides,
    "open_slide_back": common_overrides,
    "close_slide_back": common_overrides,
    "place_red_table": red_overrides,
    "place_red_drawer": red_overrides,
    "place_pink_table": pink_overrides,
    "place_pink_drawer": pink_overrides,
    "place_blue_table": blue_overrides,
    "place_blue_drawer": blue_overrides,
}


class RLegacyTapas(LegacyTapas):
    @dataclass(kw_only=True)
    class Config(LegacyTapas.Config):
        pos_reg: re.Pattern = re.compile(r"(.+?)_(?:position)")
        rot_reg: re.Pattern = re.compile(r"(.+?)_(?:rotation)")
        scalar_reg: re.Pattern = re.compile(r"(.+?)_(?:scalar)")
        area: CalvinAreaConfig = CalvinAreaConfig()
        overrides: list[str]
        name: str

        def __post_init__(self):
            self.policy.invert_prediction_batch = True

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.area = Area.create(self.cfg.area)
        self.overrides = overrides[self.cfg.name]

    def _hacky_reversed_fix(self, x: SceneObservation, y: SceneObservation) -> SceneObservation:  # type: ignore
        for override in self.overrides:
            pos = self.cfg.pos_reg.search(override)
            rot = self.cfg.rot_reg.search(override)
            scalar = self.cfg.scalar_reg.search(override)
            if override == "ee_position":
                x.ee_pose = torch.cat(
                    [
                        self.ppost[override],
                        x.ee_pose[3:],
                    ]
                )
            elif override == "ee_rotation":
                x.ee_pose = torch.cat(
                    [
                        x.ee_pose[:3],
                        self.ppost[override],
                    ]
                )
            elif override == "ee_scalar":
                x.ee_scalar = self.ppost[override]
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
                raise ValueError(f"Unknown state name: {override}")

        return x

    def _hacky_position_fix(self, x: torch.Tensor, label: str) -> torch.Tensor:
        if label in [
            "block_red_position",
            "block_pink_position",
            "block_blue_position",
        ]:
            area = self.area.check_eval_area(x)
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
