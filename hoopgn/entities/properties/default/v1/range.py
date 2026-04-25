from dataclasses import dataclass, field
from hoopgn.entities.properties.encoders.encoder import PropertyEncoder
from hoopgn.entities.properties.encoders.v1.range import RangeEncoderConfig
from hoopgn.entities.properties.evaluators.evaluator import PropertyEvaluator
from hoopgn.entities.properties.evaluators.threshold import ThresholdEvaluator
from hoopgn.entities.properties.extractors.c_gt import CGTExtractor
from hoopgn.entities.properties.extractors.extractor import PropertyExtractor
from hoopgn.entities.properties.parameters.parameter import PropertyParameter
from hoopgn.entities.properties.parameters.range import RangeParameter
from hoopgn.entities.properties.rulers.euclidean import EuclideanRuler
from hoopgn.entities.properties.rulers.ruler import PropertyRuler
from hoopgn.entities.properties.normalizers.boundary import BoundaryNormalizer

from hoopgn.entities.properties.normalizers.normalizer import PropertyNormalizer
from hoopgn.entities.properties.property import Property


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
