from dataclasses import dataclass

from hoopgn.properties.encoders.encoder import PropertyEncoder
from hoopgn.properties.encoders.v1.bool import BoolEncoder
from hoopgn.properties.evaluators.evaluator import PropertyEvaluator
from hoopgn.properties.evaluators.threshold import ThresholdEvaluator
from hoopgn.properties.normalizers.ignore import IgnoreNormalizer
from hoopgn.properties.normalizers.normalizer import PropertyNormalizer
from hoopgn.properties.parameters.binary import BinaryParameter
from hoopgn.properties.parameters.parameter import PropertyParameter
from hoopgn.properties.rulers.binary import BinaryRuler
from hoopgn.properties.rulers.ruler import PropertyRuler
from hoopgn.properties.v1 import PropertyV1


class BoolProperty(PropertyV1):
    @dataclass(kw_only=True)
    class Query(PropertyV1.Query):
        label: str = "Bool"

    @dataclass(kw_only=True)
    class Config(PropertyV1.Config):
        ruler: PropertyRuler.Config = BinaryRuler.Config()
        encoder: PropertyEncoder.Config = BoolEncoder.Config(
            query=BoolEncoder.Query(),
        )
        normalizer: PropertyNormalizer.Config = IgnoreNormalizer.Config()
        evaluator: PropertyEvaluator.Config = ThresholdEvaluator.Config()
        parameter: PropertyParameter.Config = BinaryParameter.Config()
