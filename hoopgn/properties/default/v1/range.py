from dataclasses import dataclass, field
from hoopgn.properties.encoders.encoder import PropertyEncoder
from hoopgn.properties.encoders.v1.range import RangeEncoder
from hoopgn.properties.evaluators.evaluator import PropertyEvaluator
from hoopgn.properties.evaluators.threshold import ThresholdEvaluator
from hoopgn.properties.extractors.c_gt import CGTExtractor
from hoopgn.properties.extractors.extractor import PropertyExtractor
from hoopgn.properties.parameters.parameter import PropertyParameter
from hoopgn.properties.parameters.range import RangeParameter
from hoopgn.properties.rulers.euclidean import EuclideanRuler
from hoopgn.properties.rulers.ruler import PropertyRuler
from hoopgn.properties.normalizers.boundary import BoundaryNormalizer
from hoopgn.properties.normalizers.normalizer import PropertyNormalizer
from hoopgn.properties.v1 import PropertyV1


class RangeProperty(PropertyV1):
    @dataclass(kw_only=True)
    class Query(PropertyV1.Query):
        label: str = "Range"

    @dataclass(kw_only=True)
    class Config(PropertyV1.Config):
        low: float
        high: float
        ruler: PropertyRuler.Config = EuclideanRuler.Config()
        encoder: PropertyEncoder.Config = RangeEncoder.Config(
            query=RangeEncoder.Query(),
        )
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
