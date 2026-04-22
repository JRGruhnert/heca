from dataclasses import dataclass, field
from hoopgn.networks.layers.property_encoder import PropertyEncoder
from hoopgn.properties.features.evaluators.evaluator import (
    PropertyEvaluator,
)
from hoopgn.properties.features.evaluators.threshold_evaluator import (
    ThresholdEvaluator,
)

from hoopgn.properties.features.extractors.c_gt_extractor import (
    CGTExtractor,
)
from hoopgn.properties.features.extractors.extractor import PropertyExtractor
from hoopgn.properties.features.parameters.euclidean_parameter import (
    EuclideanParameter,
)
from hoopgn.properties.features.parameters.parameter import PropertyParameter

from hoopgn.properties.features.parameters.range_parameter import RangeParameter
from hoopgn.properties.features.rulers.euclidean_ruler import (
    EuclideanRuler,
)
from hoopgn.properties.features.rulers.ruler import PropertyRuler
from hoopgn.properties.features.normalizers.boundary_normalizer import (
    BoundaryNormalizer,
)

from hoopgn.properties.features.normalizers.normalizer import (
    PropertyNormalizer,
)
from hoopgn.properties.property import Property


@dataclass(kw_only=True)
class RangeEncoderConfig(PropertyEncoder.Config):
    sig: PropertyEncoder.Signature = PropertyEncoder.Signature(
        label="Range",
    )
    dim_input: int = 1
    middle_dim: int = 8


@dataclass(kw_only=True)
class RangePropertyConfig(Property.Config):
    low: float
    high: float
    ruler: PropertyRuler.Config = EuclideanRuler.Config()
    encoder: PropertyEncoder.Config = RangeEncoderConfig()
    evaluator: PropertyEvaluator.Config = ThresholdEvaluator.Config()
    extractor: PropertyExtractor.Config = CGTExtractor.Config(field_name="Range")
    normalizer: PropertyNormalizer.Config = field(init=False)
    parameter: PropertyParameter.Config = field(init=False)

    def __post_init__(self):
        self.normalizer = BoundaryNormalizer.Config(
            lower=[self.low],
            upper=[self.high],
        )
        self.parameter = RangeParameter.Config(
            normalizer=self.normalizer,
            threshold=0.05,
        )
