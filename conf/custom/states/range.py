from dataclasses import dataclass, field
from src.states.logic.addons.prepro_scalar import ScalarStatePreprocessorConfig
from src.states.logic.addons.state_preprocessor import StatePreprocessorConfig
from src.states.logic.boundary import BoundaryConfig
from src.states.logic.values.value_linear import LinearValueConfig
from src.states.logic.threshold_boundary import BoundaryThresholdConfig
from src.states.logic.threshold_boundary import BoundaryThresholdConfig
from src.states.logic.evaluations.evaluation_threshold import ThresholdEvaluationConfig
from src.states.logic.distances.distance_binary import ScalarDistanceConfig
from src.states.logic.values.value import ValueConfig
from src.states.state import StateConfig


@dataclass
class RangeStateConfig(StateConfig):
    low: float = 0.0
    high: float = 1.0
    type_str: str = "Range"
    size: int = 1
    distance_cnd_skill: ScalarDistanceConfig = ScalarDistanceConfig()
    distance_cnd_goal: ScalarDistanceConfig = ScalarDistanceConfig()
    value_cnd_eval: ValueConfig | None = None
    eval_cnd: ThresholdEvaluationConfig = field(init=False)
    value_cnd: LinearValueConfig = field(init=False)
    preprocessor_old: StatePreprocessorConfig = field(init=False)

    def __post_init__(self):
        self.eval_cnd = ThresholdEvaluationConfig(
            distance=ScalarDistanceConfig(),
        )
        self.value_cnd = LinearValueConfig(
            boundary=BoundaryConfig(
                lower=[self.low],
                upper=[self.high],
            ),
        )
        self.preprocessor_old = ScalarStatePreprocessorConfig(
            threshold=BoundaryThresholdConfig(
                boundary=BoundaryConfig(
                    lower=[self.low],
                    upper=[self.high],
                )
            )
        )
