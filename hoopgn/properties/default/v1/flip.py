from dataclasses import dataclass

from hoopgn.properties.encoders.encoder import PropertyEncoder
from hoopgn.properties.encoders.v1.flip import FlipEncoder
from hoopgn.properties.evaluators.evaluator import PropertyEvaluator
from hoopgn.properties.evaluators.threshold import ThresholdEvaluator
from hoopgn.properties.normalizers.ignore import IgnoreNormalizer
from hoopgn.properties.normalizers.normalizer import PropertyNormalizer
from hoopgn.properties.parameters.flip import FlipParameter
from hoopgn.properties.parameters.parameter import PropertyParameter
from hoopgn.properties.rulers.flip import FlipRuler
from hoopgn.properties.rulers.ruler import PropertyRuler
from hoopgn.properties.v1 import PropertyV1


class FlipProperty(PropertyV1):
    @dataclass(kw_only=True)
    class Query(PropertyV1.Query):
        label: str = "Flip"

    @dataclass(kw_only=True)
    class Config(PropertyV1.Config):
        ruler: PropertyRuler.Config = FlipRuler.Config()
        encoder: PropertyEncoder.Config = FlipEncoder.Config(
            query=FlipEncoder.Query(),
        )
        normalizer: PropertyNormalizer.Config = IgnoreNormalizer.Config()
        evaluator: PropertyEvaluator.Config = ThresholdEvaluator.Config()
        parameter: PropertyParameter.Config = FlipParameter.Config()
