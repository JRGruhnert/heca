from dataclasses import dataclass

from src.states.logic.addons.prepro_scalar import ScalarStatePreprocessorConfig
from src.states.logic.addons.state_preprocessor import StatePreprocessorConfig
from src.states.logic.boundary import FlipBoundaryConfig
from src.states.logic.scalars.switch_distance_cnd import FlipDistanceConditionConfig
from src.states.logic.identity.identity_value_cnd import IdentityValueConfig
from src.states.logic.thresholds.threshold_boundary import BoundaryThresholdConfig
from src.states.logic.thresholds.threshold_eval_cnd import ThresholdEvalConditionConfig
from src.states.logic.scalars.range_distance_cnd import RangeDistanceConditionConfig
from src.states.logic.value_cnd import ValueConfig
from src.states.state import StateConfig


@dataclass
class FlipStateConfig(StateConfig):
    type_str: str = "Flip"
    size: int = 1
    value_cnd: IdentityValueConfig = IdentityValueConfig()
    distance_cnd_skill: FlipDistanceConditionConfig = FlipDistanceConditionConfig()
    distance_cnd_goal: RangeDistanceConditionConfig = RangeDistanceConditionConfig()
    eval_cnd: ThresholdEvalConditionConfig = ThresholdEvalConditionConfig(
        distance=RangeDistanceConditionConfig(),
    )
    value_cnd_eval: ValueConfig | None = None
    preprocessor_old: StatePreprocessorConfig = ScalarStatePreprocessorConfig(
        threshold=BoundaryThresholdConfig(
            boundary=FlipBoundaryConfig(),
        )
    )
