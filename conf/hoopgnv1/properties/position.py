from dataclasses import dataclass

from hoopgn.networks.layers.encoder import PropertyEncoderConfig
from hoopgn.entities.properties.features.evaluators.evaluator import (
    PropertyEvaluatorConfig,
)
from hoopgn.entities.properties.features.evaluators.threshold_evaluator import (
    ThresholdEvaluatorConfig,
)

from hoopgn.entities.properties.features.extractors.calvin_gt_extractor import (
    CalvinGTExtractorConfig,
)
from hoopgn.entities.properties.features.parameters.euclidean_parameter import (
    EuclideanParameterConfig,
)
from hoopgn.entities.properties.features.rulers.euclidean_ruler import (
    EuclideanRulerConfig,
)
from hoopgn.entities.properties.features.normalizers.boundary_normalizer import (
    AreaNormalizerConfig,
)
from hoopgn.entities.properties.features.conditions.condition import (
    PropertyConditionConfig,
)
from hoopgn.entities.properties.features.normalizers.normalizer import (
    PropertyNormalizerConfig,
)
from hoopgn.entities.properties.property import PropertyConfig


@dataclass
class PositionPropertyConfig(PropertyConfig):
    encoder: PropertyEncoderConfig = PropertyEncoderConfig(
        label="EulerPrecise",
        dim_input=3,
    )
    ruler: EuclideanRulerConfig = EuclideanRulerConfig()
    evaluator: PropertyEvaluatorConfig = ThresholdEvaluatorConfig(
        ruler=EuclideanRulerConfig(),
    )
    normalizer: PropertyNormalizerConfig = AreaNormalizerConfig()
    condition: PropertyConditionConfig = PropertyConditionConfig(
        ruler=EuclideanRulerConfig(),
        parameter=EuclideanParameterConfig(),
    )

    def __post_init__(self):
        self.extractor = CalvinGTExtractorConfig(label=self.label)
