from dataclasses import dataclass

from src.states.logic.addons.addon_quaternion import QuatTapasAddonConfig

from src.states.logic.rotation.quaternion_distance_cnd import (
    QuaternionDistanceConditionConfig,
)
from src.states.logic.rotation.quaternion_value_cnd import (
    QuaternionValueConditionConfig,
)
from src.states.logic.thresholds.threshold_eval_cnd import ThresholdEvalConditionConfig
from src.states.state import StateConfig


@dataclass
class QuatStateConfig(StateConfig):
    label: str = "Quat"
    size: int = 4
    ignore: bool = False
    value_cnd = QuaternionValueConditionConfig()
    distance_cnd_skill = QuaternionDistanceConditionConfig()
    distance_cnd_goal = QuaternionDistanceConditionConfig()
    eval_cnd = ThresholdEvalConditionConfig(
        distance=QuaternionDistanceConditionConfig(),
    )
    addons = {
        "tapas": QuatTapasAddonConfig(),
    }

    def __post_init__(self):
        pass
