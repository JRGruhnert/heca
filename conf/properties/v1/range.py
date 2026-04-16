from dataclasses import dataclass, field
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
from hoopgn.environments.properties.features.parameters.euclidean_parameter import (
    EuclideanParameterConfig,
)
from hoopgn.environments.properties.features.rulers.binary_ruler import (
    BinaryRulerConfig,
)
from hoopgn.environments.properties.features.rulers.euclidean_ruler import (
    EuclideanRulerConfig,
)
from hoopgn.environments.properties.features.rulers.ruler import PropertyRulerConfig
from hoopgn.environments.properties.features.normalizers.boundary_normalizer import (
    BoundaryNormalizerConfig,
)
from hoopgn.environments.properties.features.conditions.condition import (
    PropertyConditionConfig,
)
from hoopgn.environments.properties.features.normalizers.normalizer import (
    PropertyNormalizerConfig,
)
from hoopgn.environments.properties.property import PropertyConfig


@dataclass
class RangePropertyConfig(PropertyConfig):
    low: float = 0.0
    high: float = 1.0
    encoder: PropertyEncoderConfig = PropertyEncoderConfig(
        label="Range",
        dim_input=1,
        middle_dim=8,
    )
    ruler: PropertyRulerConfig = BinaryRulerConfig()
    evaluator: PropertyEvaluatorConfig = field(init=False)
    normalizer: PropertyNormalizerConfig = field(init=False)
    condition: PropertyConditionConfig = field(init=False)

    def __post_init__(self):
        self.evaluator = ThresholdEvaluatorConfig(
            ruler=BinaryRulerConfig(),
        )
        self.normalizer = BoundaryNormalizerConfig(
            lower=[self.low],
            upper=[self.high],
        )
        self.condition = PropertyConditionConfig(
            ruler=EuclideanRulerConfig(),
            parameter=EuclideanParameterConfig(),
        )
        self.extractor = CalvinGTExtractorConfig(label=self.label)
