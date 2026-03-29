from dataclasses import dataclass

from src.states.logic.flip.flip_distance_cnd import FlipDistanceConditionConfig
from src.states.logic.identity.identity_value_cnd import IdentityValueConfig
from src.states.logic.precise.precise_eval_cnd import PreciseEvalConditionConfig
from src.states.logic.range.range_distance_cnd import RangeDistanceConditionConfig
from src.states.state import StateConfig


@dataclass
class SwitchStateConfig(StateConfig):
    type_str = "Flip"
    size = 1
    value_cnd = IdentityValueConfig()
    distance_cnd_skill = FlipDistanceConditionConfig()
    distance_cnd_goal = RangeDistanceConditionConfig()
    eval_cnd = PreciseEvalConditionConfig(
        distance=RangeDistanceConditionConfig(),
    )
    addon = SwitchTapasAddonConfig()
