from dataclasses import dataclass


from hoopgn.entities.entity import Entity
from hoopgn.properties.v1 import PropertyV1
from hoopgn.environments.calvin import CalvinAreaConfig, CalvinEnvironment
from hoopgn.properties.encoders.encoder import PropertyEncoder

from hoopgn.properties.encoders.v1.area import AreaEncoder
from hoopgn.properties.parameters.parameter import PropertyParameter
from hoopgn.properties.rulers.ruler import PropertyRuler
from hoopgn.properties.evaluators.evaluator import PropertyEvaluator
from hoopgn.properties.parameters.euclidean import EuclideanParameter
from hoopgn.properties.rulers.euclidean import EuclideanRuler
from hoopgn.properties.normalizers.normalizer import PropertyNormalizer
from hoopgn.properties.evaluators.area import AreaEvaluator
from hoopgn.properties.normalizers.area import AreaNormalizer


class AreaProperty(PropertyV1):
    @dataclass(kw_only=True)
    class Query(PropertyV1.Query):
        label: str = "AreaEuler"

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
        parameter: PropertyParameter.Config = EuclideanParameter.Config()
