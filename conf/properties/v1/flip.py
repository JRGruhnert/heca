from dataclasses import dataclass

from hoopgn.networks.layers.encoder import PropertyEncoderConfig
from hoopgn.environments.properties.features.evaluators.evaluator import (
    PropertyEvaluatorConfig,
)
from hoopgn.environments.properties.features.evaluators.threshold_evaluator import (
    ThresholdEvaluatorConfig,
)

from hoopgn.environments.properties.features.extractors.calvin_gt_extractor import (
    CalvinGTExtractorConfig,
)
from hoopgn.environments.properties.features.parameters.flip_parameter import (
    FlipParameterConfig,
)
from hoopgn.environments.properties.features.rulers.binary_ruler import (
    BinaryRulerConfig,
)
from hoopgn.environments.properties.features.rulers.flip_ruler import FlipRulerConfig
from hoopgn.environments.properties.features.rulers.ruler import PropertyRulerConfig
from hoopgn.environments.properties.features.normalizers.boundary_normalizer import (
    BoolNormalizerConfig,
)
from hoopgn.environments.properties.features.conditions.condition import (
    PropertyConditionConfig,
)
from hoopgn.environments.properties.features.normalizers.normalizer import (
    PropertyNormalizerConfig,
)
from hoopgn.environments.properties.property import PropertyConfig


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
