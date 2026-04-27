from dataclasses import dataclass

from heca.properties.encoders.encoder import PropertyEncoder

from heca.properties.encoders.v2.position import PositionEncoder
from heca.properties.evaluators.evaluator import PropertyEvaluator
from heca.properties.evaluators.threshold import ThresholdEvaluator

from heca.properties.extractors.extractor import PropertyExtractor
from heca.properties.normalizers.normalizer import PropertyNormalizer
from heca.properties.parameters.euclidean import EuclideanParameter
from heca.properties.parameters.parameter import PropertyParameter
from heca.properties.rulers.euclidean import EuclideanRuler
from heca.properties.normalizers.area import AreaNormalizer

from heca.properties.rulers.ruler import PropertyRuler
from heca.properties.property import Property


class PositionProperty(Property):
    @dataclass(kw_only=True)
    class Query(Property.Query):
        label: str = "position"

    @dataclass(kw_only=True)
    class Config(Property.Config):
        ruler: PropertyRuler.Config = EuclideanRuler.Config()
        encoder: PropertyEncoder.Config = PositionEncoder.Config(
            query=PositionEncoder.Query(),
        )
        evaluator: PropertyEvaluator.Config = ThresholdEvaluator.Config()
        normalizer: PropertyNormalizer.Config = AreaNormalizer.Config()
        parameter: PropertyParameter.Config = EuclideanParameter.Config()
