from dataclasses import dataclass

from src.networks.layers.encoder import StateEncoderConfig
from src.states.logic.addons.prepro_euclidean import EuclideanStatePreprocessorConfig
from src.states.logic.value_handler.normalizers.boundary_normalizer import (
    AreaBoundaryConfig,
)
from src.states.logic.condition import ConditionConfig
from src.states.logic.values.value_linear import LinearValueConfig
from src.states.logic.distances.distance_euclidean import (
    EuclideanDistanceConfig,
)

from src.states.logic.evaluations.evaluation_threshold import ThresholdEvaluationConfig
from src.states.logic.values.value_handler import ValueHandler
from src.states.state import StateConfig


@dataclass
class PositionStateConfig(StateConfig):
    encoder: StateEncoderConfig = StateEncoderConfig(
        label="EulerPrecise",
        dim_input=3,
    )
    distance_goal: EuclideanDistanceConfig = EuclideanDistanceConfig()
    distance_skill: EuclideanDistanceConfig = EuclideanDistanceConfig()
    eval_handler: ThresholdEvaluationConfig = ThresholdEvaluationConfig(
        distance=EuclideanDistanceConfig(),
    )
    value_handler: LinearValueConfig = LinearValueConfig(
        boundary=AreaBoundaryConfig(),
    )
    condition: ConditionConfig = ConditionConfig(
        distance=EuclideanDistanceConfig(),
        preprocessor=EuclideanStatePreprocessorConfig(),
    )
    value_handler_eval: ValueHandler | None = None
