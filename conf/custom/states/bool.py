from dataclasses import dataclass

from src.networks.layers.encoder import StateEncoderConfig
from src.states.logic.addons.prepro_scalar import ScalarStatePreprocessorConfig
from src.states.logic.distances.distance import DistanceConfig
from src.states.logic.value_handler.normalizers.boundary_normalizer import (
    BoolBoundaryConfig,
)
from src.states.logic.condition import ConditionConfig
from src.states.logic.value_handler.normalizers.ignore_normalizer import (
    IgnoreValueConfig,
)
from src.states.logic.value_handler.normalizers.normalizer import NormalizerConfig
from src.states.logic.threshold_boundary import BoundaryThresholdConfig
from src.states.logic.evaluations.evaluation_threshold import ThresholdEvaluationConfig
from src.states.logic.distances.distance_binary import ScalarDistanceConfig
from src.states.state import StateConfig


@dataclass
class BoolStateConfig(StateConfig):
    encoder: StateEncoderConfig = StateEncoderConfig(
        label="Bool",
        dim_input=1,
        middle_dim=8,
    )
    normalizer: NormalizerConfig = IgnoreValueConfig()
    distance: DistanceConfig = ScalarDistanceConfig()
    evaluator: ThresholdEvaluationConfig = ThresholdEvaluationConfig(
        distance=ScalarDistanceConfig(),
    )
    condition: ConditionConfig = ConditionConfig(
        distance=ScalarDistanceConfig(),
        preprocessor=ScalarStatePreprocessorConfig(
            threshold=BoundaryThresholdConfig(
                boundary=BoolBoundaryConfig(),
            )
        ),
    )
