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
from hoopgn.environments.properties.features.parameters.binary_parameter import (
    BinaryParameterConfig,
)
from hoopgn.environments.properties.features.rulers.binary_ruler import (
    BinaryRulerConfig,
)
from hoopgn.environments.properties.features.rulers.ruler import PropertyRulerConfig

from hoopgn.environments.properties.features.normalizers.ignore_normalizer import (
    IgnoreNormalizerConfig,
)
from hoopgn.environments.properties.features.normalizers.normalizer import (
    PropertyNormalizerConfig,
)
from hoopgn.environments.properties.property import PropertyConfig
from hoopgn.environments.properties.features.conditions.condition import (
    PropertyConditionConfig,
)


@dataclass
class BoolPropertyConfig(PropertyConfig):
    encoder: PropertyEncoderConfig = PropertyEncoderConfig(
        label="Bool",
        dim_input=1,
        middle_dim=8,
    )
    normalizer: PropertyNormalizerConfig = IgnoreNormalizerConfig()
    ruler: PropertyRulerConfig = BinaryRulerConfig()
    evaluator: PropertyEvaluatorConfig = ThresholdEvaluatorConfig(
        ruler=BinaryRulerConfig(),
    )
    condition: PropertyConditionConfig = PropertyConditionConfig(
        ruler=BinaryRulerConfig(),
        parameter=BinaryParameterConfig(),
    )

    def __post_init__(self):
        self.extractor = CalvinGTExtractorConfig(label=self.label)
