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
    ruler: PropertyRuler.Config = BinaryRuler.Config()
    evaluator: PropertyEvaluator.Config = field(init=False)
    normalizer: PropertyNormalizer.Config = field(init=False)
    condition: PropertyCondition.Config = field(init=False)

    def __post_init__(self):
        self.evaluator = ThresholdEvaluator.Config(
            ruler=BinaryRuler.Config(),
        )
        self.normalizer = BoundaryNormalizer.Config(
            lower=[self.low],
            upper=[self.high],
        )
        self.condition = PropertyCondition.Config(
            ruler=EuclideanRuler.Config(),
            parameter=EuclideanParameter.Config(),
        )
        self.extractor = CalvinGTExtractor.Config(label=self.label)
