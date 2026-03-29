from dataclasses import dataclass

from src.states.logic.precise.precise_eval_cnd import PreciseEvalConditionConfig
from src.states.logic.rotation.quaternion import QuaternionConfig
from src.states.logic.rotation.quaternion_distance_cnd import (
    QuaternionDistanceConditionConfig,
)
from src.states.logic.rotation.quaternion_value_cnd import (
    QuaternionValueConditionConfig,
)
from src.states.state import State, StateConfig


@dataclass
class QuatStateConfig(StateConfig):
    label: str = "Quat"
    size: int = 4
    value_cnd = QuaternionValueConditionConfig()
    distance_cnd_skill = QuaternionDistanceConditionConfig()
    distance_cnd_goal = QuaternionDistanceConditionConfig()
    eval_cnd = PreciseEvalConditionConfig(
        distance=QuaternionDistanceConditionConfig(),
    )
    addons = {
        "tapas": QuatTapasAddonConfig(),
    }


class QuatState(State):
    def __init__(
        self,
        name: str,
        id: int,
        ignore: bool = False,
    ):
        super().__init__(
            name=name,
            id=id,
            type_str="Quat",
            size=4,
            value_condition=QuaternionValueNormalizer(),
            skill_condition=QuaternionDistanceCondition(),
            goal_condition=QuaternionDistanceCondition(),
            eval_condition=(
                IgnoreEvalCondition()
                if ignore
                else PreciseEvalCondition(condition=QuaternionDistanceCondition())
            ),
            addon=QuatTapasAddon(),
        )
