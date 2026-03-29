from attr import dataclass
from src.skills.addons.addon_scalar import ScalarTapasAddonConfig
from src.states.logic.boundary import BoundaryConfig
from src.states.logic.linear.linear_value_cnd import LinearValueNormalizerConfig
from src.states.logic.precise.precise_eval_cnd import PreciseEvalConditionConfig
from src.states.logic.range.range_distance_cnd import RangeDistanceConditionConfig
from src.states.state import StateConfig


@dataclass
class RangeStateConfig(StateConfig):
    lower_bound: float
    upper_bound: float
    label: str = "Range"
    size: int = 1
    value_cnd = LinearValueNormalizerConfig(
        boundary=BoundaryConfig(
            lower_bound=[0.0],
            upper_bound=[1.0],
        ),
    )
    eval_cnd = PreciseEvalConditionConfig(
        distance=RangeDistanceConditionConfig(),
    )
    eval_normalizer = None
    addons = {
        "tapas": ScalarTapasAddonConfig(
            lower_bound=[0.0],
            upper_bound=[1.0],
        ),
    }
