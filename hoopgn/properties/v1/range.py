from dataclasses import dataclass, field
from hoopgn.networks.layers.property_encoder import PropertyEncoder
from hoopgn.properties.features.evaluators.evaluator import (
    PropertyEvaluator,
)
from hoopgn.properties.features.evaluators.threshold_evaluator import (
    ThresholdEvaluator,
)

from hoopgn.properties.features.extractors.calvin_gt_extractor import (
    CalvinGTExtractor,
)
from hoopgn.properties.features.parameters.euclidean_parameter import (
    EuclideanParameter,
)
from hoopgn.properties.features.parameters.parameter import PropertyParameter
from hoopgn.properties.features.rulers.binary_ruler import (
    BinaryRuler,
)
from hoopgn.properties.features.rulers.euclidean_ruler import (
    EuclideanRuler,
)
from hoopgn.properties.features.rulers.ruler import PropertyRuler
from hoopgn.properties.features.normalizers.boundary_normalizer import (
    BoundaryNormalizer,
)
from hoopgn.properties.features.conditions.condition import (
    PropertyCondition,
)
from hoopgn.properties.features.normalizers.normalizer import (
    PropertyNormalizer,
)
from hoopgn.properties.property import Property


@dataclass
class RangePropertyConfig(Property.Config):
    low: float = 0.0
    high: float = 1.0
    encoder: PropertyEncoder.Config = PropertyEncoder.Config(
        label="Range",
        dim_input=1,
        middle_dim=8,
    )
    ruler: PropertyRuler.Config = EuclideanRuler.Config()
    evaluator = ThresholdEvaluator.Config(
        ruler=EuclideanRuler.Config(),
    )
    parameter: PropertyParameter.Config = EuclideanParameter.Config()
    normalizer: PropertyNormalizer.Config = field(init=False)

    def __post_init__(self):
        self.evaluator = ThresholdEvaluator.Config(
            ruler=EuclideanRuler.Config(),
        )
        self.normalizer = BoundaryNormalizer.Config(
            lower=[self.low],
            upper=[self.high],
        )
        self.extractor = CalvinGTExtractor.Config(field_name=self.label)
