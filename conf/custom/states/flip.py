from dataclasses import dataclass

from src.networks.layers.encoder import StateEncoderConfig
from src.states.logic.addons.prepro_scalar import ScalarStatePreprocessorConfig
from src.states.logic.boundary import FlipBoundaryConfig
from src.states.logic.condition import ConditionConfig
from src.states.logic.distances.distance_flip_special import FlipDistanceConfig
from src.states.logic.values.value_identity import IdentityValueConfig
from src.states.logic.threshold_boundary import BoundaryThresholdConfig
from src.states.logic.evaluations.evaluation_threshold import ThresholdEvaluationConfig
from src.states.logic.distances.distance_binary import ScalarDistanceConfig
from src.states.logic.values.value import ValueHandlerConfig
from src.states.state import StateConfig


@dataclass
class FlipStateConfig(StateConfig):
    encoder: StateEncoderConfig = StateEncoderConfig(
        label="Flip",
        dim_input=1,
        middle_dim=8,
    )
    value_handler: IdentityValueConfig = IdentityValueConfig()
    distance_skill: FlipDistanceConfig = FlipDistanceConfig()
    distance_goal: ScalarDistanceConfig = ScalarDistanceConfig()
    eval_handler: ThresholdEvaluationConfig = ThresholdEvaluationConfig(
        distance=ScalarDistanceConfig(),
    )
    value_handler_eval: ValueHandlerConfig | None = None
    condition: ConditionConfig = ConditionConfig(
        distance=FlipDistanceConfig(),
        preprocessor=ScalarStatePreprocessorConfig(
            threshold=BoundaryThresholdConfig(
                boundary=FlipBoundaryConfig(),
            )
        ),
    )
