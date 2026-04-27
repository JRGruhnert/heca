from dataclasses import dataclass

from heca.properties.encoders.encoder import PropertyEncoder
from heca.properties.encoders.v1.position import PositionEncoder
from heca.properties.evaluators.evaluator import PropertyEvaluator
from heca.properties.evaluators.threshold import ThresholdEvaluator
from heca.properties.normalizers.normalizer import PropertyNormalizer
from heca.properties.parameters.euclidean import EuclideanParameter
from heca.properties.parameters.parameter import PropertyParameter
from heca.properties.rulers.euclidean import EuclideanRuler
from heca.properties.normalizers.area import AreaNormalizer
from heca.properties.rulers.ruler import PropertyRuler
from heca.properties.v1 import PropertyV1


class PositionProperty(PropertyV1):
    @dataclass(kw_only=True)
    class Config(PropertyV1.Config):
        ruler: PropertyRuler.Config = EuclideanRuler.Config()
        encoder: PropertyEncoder.Config = PositionEncoder.Config(
            query=PositionEncoder.Query(),
        )
        evaluator: PropertyEvaluator.Config = ThresholdEvaluator.Config()
        normalizer: PropertyNormalizer.Config = AreaNormalizer.Config()
        parameter: PropertyParameter.Config = EuclideanParameter.Config()
