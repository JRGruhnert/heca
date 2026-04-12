from dataclasses import dataclass

from hoopgn.networks.layers.encoder import PropertyEncoderConfig
from hoopgn.properties.features.evaluators.evaluator import (
    PropertyEvaluatorConfig,
)
from hoopgn.properties.features.evaluators.threshold_evaluator import (
    ThresholdEvaluatorConfig,
)

from hoopgn.properties.features.extractors.calvin_gt_extractor import (
    CalvinGTExtractorConfig,
)
from hoopgn.properties.features.parameters.flip_parameter import (
    FlipParameterConfig,
)
from hoopgn.properties.features.rulers.binary_ruler import BinaryRulerConfig
from hoopgn.properties.features.rulers.flip_ruler import FlipRulerConfig
from hoopgn.properties.features.rulers.ruler import PropertyRulerConfig
from hoopgn.properties.features.normalizers.boundary_normalizer import (
    BoolNormalizerConfig,
)
from hoopgn.properties.features.conditions.condition import (
    PropertyConditionConfig,
)
from hoopgn.properties.features.normalizers.normalizer import (
    PropertyNormalizerConfig,
)
from hoopgn.properties.property import PropertyConfig


@dataclass
class FlipPropertyConfig(PropertyConfig):
    encoder: PropertyEncoderConfig = PropertyEncoderConfig(
        label="Flip",
        dim_input=1,
        middle_dim=8,
    )
    normalizer: PropertyNormalizerConfig = BoolNormalizerConfig()
    ruler: PropertyRulerConfig = BinaryRulerConfig()
    evaluator: PropertyEvaluatorConfig = ThresholdEvaluatorConfig(
        ruler=BinaryRulerConfig(),
    )
    condition: PropertyConditionConfig = PropertyConditionConfig(
        ruler=FlipRulerConfig(),
        parameter=FlipParameterConfig(),
    )

    def __post_init__(self):
        self.extractor = CalvinGTExtractorConfig(label=self.label)
