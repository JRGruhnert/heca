from dataclasses import dataclass

from src.networks.layers.encoder import StateEncoderConfig
from src.objects.properties.value_handler.evaluators.evaluator import (
    StateEvaluatorConfig,
)
from src.objects.properties.value_handler.evaluators.threshold_evaluator import (
    ThresholdEvaluatorConfig,
)
from src.objects.properties.value_handler.parameters.flip_parameter import (
    FlipParameterConfig,
)
from src.objects.properties.value_handler.rulers.binary_ruler import BinaryRulerConfig
from src.objects.properties.value_handler.rulers.flip_ruler import FlipRulerConfig
from src.objects.properties.value_handler.rulers.ruler import RulerConfig
from src.objects.properties.value_handler.normalizers.boundary_normalizer import (
    BoolBoundaryConfig,
)
from src.objects.properties.condition import ConditionConfig
from src.objects.properties.value_handler.normalizers.normalizer import NormalizerConfig
from src.objects.properties.property import StateConfig


@dataclass
class FlipStateConfig(StateConfig):
    encoder: StateEncoderConfig = StateEncoderConfig(
        label="Flip",
        dim_input=1,
        middle_dim=8,
    )
    normalizer: NormalizerConfig = BoolBoundaryConfig()
    ruler: RulerConfig = BinaryRulerConfig()
    evaluator: StateEvaluatorConfig = ThresholdEvaluatorConfig(
        ruler=BinaryRulerConfig(),
    )
    condition: ConditionConfig = ConditionConfig(
        ruler=FlipRulerConfig(),
        parameter=FlipParameterConfig(),
    )
