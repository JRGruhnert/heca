from dataclasses import dataclass

from heca.properties.v1 import PropertyV1
from heca.properties.encoders.encoder import PropertyEncoder
from heca.properties.encoders.v1.area import AreaEncoder
from heca.properties.rulers.ruler import PropertyRuler
from heca.properties.evaluators.evaluator import PropertyEvaluator
from heca.properties.rulers.euclidean import EuclideanRuler
from heca.properties.normalizers.normalizer import PropertyNormalizer
from heca.properties.evaluators.area import AreaEvaluator
from heca.properties.normalizers.area import AreaNormalizer
from heca.environments.scenes.calvin.area import CalvinAreaConfig


class AreaProperty(PropertyV1):
    @dataclass(kw_only=True)
    class Config(PropertyV1.Config):
        ruler: PropertyRuler.Config = EuclideanRuler.Config()
        encoder: PropertyEncoder.Config = AreaEncoder.Config(
            query=AreaEncoder.Query(),
        )
        normalizer: PropertyNormalizer.Config = AreaNormalizer.Config()
        evaluator: PropertyEvaluator.Config = AreaEvaluator.Config(
            area=CalvinAreaConfig(),
        )
