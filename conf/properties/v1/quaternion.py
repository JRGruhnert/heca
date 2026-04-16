from dataclasses import dataclass

from hoopgn.networks.layers.encoder import PropertyEncoderConfig
from hoopgn.environments.properties.features.evaluators.evaluator import (
    PropertyEvaluatorConfig,
)
from hoopgn.environments.properties.features.evaluators.threshold_evaluator import (
    ThresholdEvaluatorConfig,
)
from hoopgn.environments.properties.features.conditions.condition import (
    PropertyConditionConfig,
)

from hoopgn.environments.properties.features.extractors.calvin_gt_extractor import (
    CalvinGTExtractorConfig,
)
from hoopgn.environments.properties.features.parameters.quaternion_parameter import (
    QuaternionParameterConfig,
)
from hoopgn.environments.properties.features.rulers.angular_ruler import (
    AngularRulerConfig,
)
from hoopgn.environments.properties.features.normalizers.quaternion_normalizer import (
    QuaternionNormalizerConfig,
)
from hoopgn.environments.properties.property import PropertyConfig


@dataclass
class QuaternionPropertyConfig(PropertyConfig):
    encoder: PropertyEncoderConfig = PropertyEncoderConfig(
        label="Quat",
        dim_input=4,
    )
    normalizer: QuaternionNormalizerConfig = QuaternionNormalizerConfig()
    ruler: AngularRulerConfig = AngularRulerConfig()
    evaluator: PropertyEvaluatorConfig = ThresholdEvaluatorConfig(
        ruler=AngularRulerConfig(),
    )
    condition: PropertyConditionConfig = PropertyConditionConfig(
        ruler=AngularRulerConfig(),
        parameter=QuaternionParameterConfig(),
    )

    def __post_init__(self):
        self.extractor = CalvinGTExtractorConfig(label=self.label)
