from dataclasses import dataclass, field
from src.states.logic.addons.addon_scalar import ScalarStatePreprocessorConfig
from src.states.logic.boundary import BoundaryConfig
from src.states.logic.linear.linear_value_cnd import LinearValueNormalizerConfig
from src.states.logic.thresholds.threshold_boundary import BoundaryThresholdConfig
from src.states.logic.thresholds.threshold_boundary import BoundaryThresholdConfig
from src.states.logic.thresholds.threshold_eval_cnd import ThresholdEvalConditionConfig
from src.states.logic.scalars.range_distance_cnd import RangeDistanceConditionConfig
from src.states.logic.value_cnd import ValueConditionConfig
from src.states.state import StateConfig


@dataclass
class RangeStateConfig(StateConfig):
    low: float = 0.0
    high: float = 1.0
    type_str: str = "Range"
    size: int = 1
    distance_cnd_skill: RangeDistanceConditionConfig = RangeDistanceConditionConfig()
    distance_cnd_goal: RangeDistanceConditionConfig = RangeDistanceConditionConfig()
    value_cnd_eval: ValueConditionConfig | None = None
    eval_cnd: ThresholdEvalConditionConfig = field(init=False)
    value_cnd: LinearValueNormalizerConfig = field(init=False)
    addons: dict[str, ScalarStatePreprocessorConfig] = field(init=False)

    def __post_init__(self):
        self.eval_cnd = ThresholdEvalConditionConfig(
            distance=RangeDistanceConditionConfig(),
        )
        self.value_cnd = LinearValueNormalizerConfig(
            boundary=BoundaryConfig(
                lower=[self.low],
                upper=[self.high],
            ),
        )
        self.addons = {
            "tapas": ScalarStatePreprocessorConfig(
                threshold=BoundaryThresholdConfig(
                    boundary=BoundaryConfig(
                        lower=[self.low],
                        upper=[self.high],
                    )
                )
            ),
        }
