from dataclasses import dataclass

from src.networks.layers.encoder import StateEncoderConfig
from src.states.logic.addons.prepro_scalar import ScalarStatePreprocessorConfig
from src.states.logic.value_handler.normalizers.boundary_normalizer import (
    BoolBoundaryConfig,
)
from src.states.logic.condition import ConditionConfig
from src.states.logic.distances.distance_flip_special import FlipDistanceConfig
from src.states.logic.threshold_boundary import BoundaryThresholdConfig
from src.states.logic.evaluations.evaluation_threshold import ThresholdEvaluationConfig
from src.states.logic.distances.distance_binary import ScalarDistanceConfig
from src.states.logic.value_handler.normalizers.normalizer import NormalizerConfig
from src.states.state import ObjectConfig


@dataclass
class FlipStateConfig(ObjectConfig):
    encoder: StateEncoderConfig = StateEncoderConfig(
        label="Flip",
        dim_input=1,
        middle_dim=8,
    )
    normalizer: NormalizerConfig = BoolBoundaryConfig()
    distance: ScalarDistanceConfig = ScalarDistanceConfig()
    evaluator: ThresholdEvaluationConfig = ThresholdEvaluationConfig(
        distance=ScalarDistanceConfig(),
    )
    condition: ConditionConfig = ConditionConfig(
        distance=FlipDistanceConfig(),
        preprocessor=ScalarStatePreprocessorConfig(
            threshold=BoundaryThresholdConfig(
                boundary=BoolBoundaryConfig(),
            )
        ),
    )
