from dataclasses import dataclass

from src.states.logic.addons.prepro_scalar import ScalarStatePreprocessorConfig
from src.states.logic.boundary import BoundaryConfig, SwitchBoundaryConfig
from src.states.logic.identity.identity_value_cnd import IdentityValueConfig
from src.states.logic.thresholds.threshold_boundary import BoundaryThresholdConfig
from src.states.logic.thresholds.threshold_eval_cnd import ThresholdEvalConditionConfig
from src.states.logic.scalars.range_distance_cnd import RangeDistanceConditionConfig
from src.states.logic.value_cnd import ValueConditionConfig
from src.states.state import StateConfig


@dataclass
class BoolStateConfig(StateConfig):
    type_str: str = "Bool"
    size: int = 1
    value_cnd: IdentityValueConfig = IdentityValueConfig()
    distance_cnd_skill: RangeDistanceConditionConfig = RangeDistanceConditionConfig()
    distance_cnd_goal: RangeDistanceConditionConfig = RangeDistanceConditionConfig()
    eval_cnd: ThresholdEvalConditionConfig = ThresholdEvalConditionConfig(
        distance=RangeDistanceConditionConfig(),
    )
    value_cnd_eval: ValueConditionConfig | None = None

    addons: dict[str, ScalarStatePreprocessorConfig] = {
        "tapas": ScalarStatePreprocessorConfig(
            threshold=BoundaryThresholdConfig(
                boundary=SwitchBoundaryConfig(),
            )
        ),
    }
