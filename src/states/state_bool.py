from dataclasses import dataclass

from src.states.logic.addons.addon_scalar import ScalarTapasAddonConfig
from src.states.logic.boundary import BoundaryConfig
from src.states.logic.identity.identity_value_cnd import IdentityValueConfig
from src.states.logic.thresholds.threshold_boundary import BoundaryThresholdConfig
from src.states.logic.thresholds.threshold_eval_cnd import ThresholdEvalConditionConfig
from src.states.logic.scalars.range_distance_cnd import RangeDistanceConditionConfig
from src.states.logic.value_cnd import ValueConditionConfig
from src.states.state import StateConfig


@dataclass
class BoolStateConfig(StateConfig):
    label = "Bool"
    size = 1
    value_cnd = IdentityValueConfig()
    distance_cnd_skill = RangeDistanceConditionConfig()
    distance_cnd_goal = RangeDistanceConditionConfig()
    eval_cnd = ThresholdEvalConditionConfig(
        distance=RangeDistanceConditionConfig(),
    )
    value_cnd_eval: ValueConditionConfig | None = None

    addons = {
        "tapas": ScalarTapasAddonConfig(
            threshold=BoundaryThresholdConfig(
                boundary=BoundaryConfig(
                    lower_bound=[0.0],
                    upper_bound=[1.0],
                )
            )
        ),
    }
