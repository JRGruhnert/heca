from dataclasses import dataclass, field
from src.networks.layers.encoder import StateEncoderConfig
from src.states.logic.addons.prepro_scalar import ScalarStatePreprocessorConfig
from src.states.logic.boundary import BoundaryConfig
from src.states.logic.condition import ConditionConfig
from src.states.logic.distances.distance_euclidean import EuclideanDistanceConfig
from src.states.logic.values.value_linear import LinearValueConfig
from src.states.logic.threshold_boundary import BoundaryThresholdConfig
from src.states.logic.threshold_boundary import BoundaryThresholdConfig
from src.states.logic.evaluations.evaluation_threshold import ThresholdEvaluationConfig
from src.states.logic.distances.distance_binary import ScalarDistanceConfig
from src.states.logic.values.value import ValueHandlerConfig
from src.states.state import StateConfig


@dataclass
class RangeStateConfig(StateConfig):
    low: float = 0.0
    high: float = 1.0
    encoder: StateEncoderConfig = StateEncoderConfig(
        label="Range",
        dim_input=1,
        middle_dim=8,
    )
    distance_skill: ScalarDistanceConfig = ScalarDistanceConfig()
    distance_goal: ScalarDistanceConfig = ScalarDistanceConfig()
    value_handler_eval: ValueHandlerConfig | None = None
    eval_handler: ThresholdEvaluationConfig = field(init=False)
    value_handler: LinearValueConfig = field(init=False)
    condition: ConditionConfig = field(init=False)

    def __post_init__(self):
        self.eval_handler = ThresholdEvaluationConfig(
            distance=ScalarDistanceConfig(),
        )
        boundary = BoundaryConfig(
            lower=[self.low],
            upper=[self.high],
        )

        self.value_handler = LinearValueConfig(boundary=boundary)
        self.condition = ConditionConfig(
            distance=EuclideanDistanceConfig(),
            preprocessor=ScalarStatePreprocessorConfig(
                threshold=BoundaryThresholdConfig(boundary=boundary),
            ),
        )
