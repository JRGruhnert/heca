from dataclasses import dataclass, field
from heca.properties.encoders.encoder import PropertyEncoder
from heca.properties.encoders.v1.range import RangeEncoder
from heca.properties.evaluators.evaluator import PropertyEvaluator
from heca.properties.evaluators.threshold import ThresholdEvaluator
from heca.properties.extractors.gt import CGTExtractor
from heca.properties.extractors.extractor import PropertyExtractor
from heca.properties.parameters.parameter import PropertyParameter
from heca.properties.parameters.range import RangeParameter
from heca.properties.rulers.euclidean import EuclideanRuler
from heca.properties.rulers.ruler import PropertyRuler
from heca.properties.normalizers.boundary import BoundaryNormalizer
from heca.properties.normalizers.normalizer import PropertyNormalizer
from heca.properties.v1 import PropertyV1


class RangeProperty(PropertyV1):
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
