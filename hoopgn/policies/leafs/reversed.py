from dataclasses import dataclass
from functools import cached_property
import re
import sys

from hoopgn.misc import logger
import numpy as np
import torch

from hoopgn.policies.leafs.tapas import TapasPolicy
from hoopgn.properties.property import Property
from hoopgn.observation.td_properties import TDProperties
from hoopgn.properties.evaluators.area import AreaEvaluator
from hoopgn.properties.default.v1.area import CalvinAreaConfig

import tapas_gmm_modified
from tapas_gmm_modified.utils.observation import SceneObservation

sys.modules["tapas_gmm"] = tapas_gmm_modified  # alias for unpickling old checkpoints


class RTapasPolicy(TapasPolicy):
    @dataclass(kw_only=True)
    class Config(TapasPolicy.Config):
        overrides: set[Property.Signature]
        pos_reg: re.Pattern = re.compile(r"(.+?)_(?:position)")
        rot_reg: re.Pattern = re.compile(r"(.+?)_(?:rotation)")
        scalar_reg: re.Pattern = re.compile(r"(.+?)_(?:scalar)")

        def __post_init__(self):
            self.tapas.invert_prediction_batch = True

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.evaluator = AreaEvaluator(AreaEvaluator.Config(area=CalvinAreaConfig()))

    def get_value(self, x: TDProperties) -> torch.Tensor:
        assert "tapas" in x.keys(), "Tapas observation must be present in the input."
        return self._hacky_reversed_fix(x["tapas"])

    def _hacky_reversed_fix(self, obs: SceneObservation) -> SceneObservation:  # type: ignore
        assert self.goal is not None, "Goal must be set for hacky postprocess."
        for sig in self.cfg.overrides:
            pos = self.cfg.pos_reg.search(sig.label)
            rot = self.cfg.rot_reg.search(sig.label)
            scalar = self.cfg.scalar_reg.search(sig.label)
            if sig.label == "ee_position":
                obs.ee_pose = torch.cat(
                    [
                        self.ppost[sig],
                        obs.ee_pose[3:],
                    ]
                )
            elif sig.label == "ee_rotation":
                obs.ee_pose = torch.cat(
                    [
                        obs.ee_pose[:3],
                        self.ppost[sig],
                    ]
                )
            elif sig.label == "ee_scalar":
                obs.ee_scalar = self.ppost[sig]
            elif pos:
                obs.object_poses[pos.group(1)] = np.concatenate(
                    [
                        self._hacky_position_fix(pos.group(1)),
                        obs.object_poses[pos.group(1)][3:],
                    ]
                )
            elif rot:
                obs.object_poses[rot.group(1)] = np.concatenate(
                    [
                        obs.object_poses[rot.group(1)][:3],
                        self.goal[f"{rot.group(1)}_rotation"].numpy(),
                    ]
                )
            elif scalar:
                obs.object_states[scalar.group(1)] = self.goal[
                    f"{scalar.group(1)}_scalar"
                ].numpy()
            else:
                raise ValueError(f"Unknown state name: {sig.label}")

        return obs

    def _hacky_position_fix(self, label: str) -> np.ndarray:
        assert self.goal is not None, "Goal must be set for hacky position."
        if label in [  # TODO: check if names correct
            "block_red_position",
            "block_pink_position",
            "block_blue_position",
        ]:
            area = self.evaluator.area.check_eval_area(self.goal[label])
            pos = self.goal[label].clone()
            if (
                area == "drawer_closed"
            ):  # NOTE: This is a hardcoded fix for the drawer, since the position of the drawer in the reversed trajectory is in the closed position but the model expects it to be in the open position. This is because the model was trained with the open position as precondition and the closed position as postcondition, so when we reverse it, we need to also reverse the positions.
                pos[1] -= 0.17  # Drawer Offset
        else:
            pos = self.goal[label]
        return pos.numpy()

    @cached_property
    def overrides(self) -> dict[Property.Signature, torch.Tensor]:
        values = self._load_conditions(False)
        for k, v in values.items():
            if k.label in self.cfg.overrides:
                logger.info(f"Overriding {k.label} with value {v.cpu().numpy()}.")
        return values

    @cached_property
    def demo_precons(self) -> dict[str, torch.Tensor]:
        return self.model.end_values

    @cached_property
    def demo_postcons(self) -> dict[str, torch.Tensor]:
        return self.model.start_values
