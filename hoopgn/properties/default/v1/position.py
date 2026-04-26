from dataclasses import dataclass

from hoopgn.properties.encoders.encoder import PropertyEncoder
from hoopgn.properties.encoders.v1.position import PositionEncoder
from hoopgn.properties.evaluators.evaluator import PropertyEvaluator
from hoopgn.properties.evaluators.threshold import ThresholdEvaluator
from hoopgn.properties.normalizers.normalizer import PropertyNormalizer
from hoopgn.properties.parameters.euclidean import EuclideanParameter
from hoopgn.properties.parameters.parameter import PropertyParameter
from hoopgn.properties.rulers.euclidean import EuclideanRuler
from hoopgn.properties.normalizers.area import AreaNormalizer
from hoopgn.properties.rulers.ruler import PropertyRuler
from hoopgn.properties.property import Property
from hoopgn.properties.v1 import PropertyV1


class PositionProperty(PropertyV1):
    @dataclass(kw_only=True)
    class Query(PropertyV1.Query):
        label: str = "EulerPrecise"

    @dataclass(kw_only=True)
    class Config(PropertyV1.Config):
        ruler: PropertyRuler.Config = EuclideanRuler.Config()
        encoder: PropertyEncoder.Config = PositionEncoder.Config(
            query=PositionEncoder.Query(),
        )
        evaluator: PropertyEvaluator.Config = ThresholdEvaluator.Config()
        normalizer: PropertyNormalizer.Config = AreaNormalizer.Config()
        parameter: PropertyParameter.Config = EuclideanParameter.Config()
