from dataclasses import dataclass

from src.skills.addons.addon_scalar import ScalarTapasAddonConfig
from src.states.logic.identity.identity_value_cnd import IdentityValueConfig
from src.states.logic.precise.precise_eval_cnd import PreciseEvalConditionConfig
from src.states.logic.range.range_distance_cnd import RangeDistanceConditionConfig
from src.states.state import StateConfig


@dataclass
class BoolStateConfig(StateConfig):
    type_str = "Bool"
    size = 1
    value_cnd = IdentityValueConfig()
    distance_cnd_skill = RangeDistanceConditionConfig()
    distance_cnd_goal = RangeDistanceConditionConfig()
    eval_cnd = PreciseEvalConditionConfig(
        distance=RangeDistanceConditionConfig(),
    )
    addon = ScalarTapasAddonConfig(
        lower_bound=[0.0],
        upper_bound=[1.0],
    )
