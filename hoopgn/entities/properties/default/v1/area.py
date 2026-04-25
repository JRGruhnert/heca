from dataclasses import dataclass


from hoopgn.environments.calvin import CalvinAreaConfig
from hoopgn.entities.properties.encoders.encoder import PropertyEncoder

from hoopgn.entities.properties.encoders.v1.area import AreaEncoderConfig
from hoopgn.entities.properties.extractors.c_gt_area import CGTAreaExtractor
from hoopgn.entities.properties.extractors.extractor import PropertyExtractor
from hoopgn.entities.properties.parameters.parameter import PropertyParameter
from hoopgn.entities.properties.rulers.ruler import PropertyRuler
from hoopgn.entities.properties.evaluators.evaluator import PropertyEvaluator
from hoopgn.entities.properties.parameters.euclidean import EuclideanParameter
from hoopgn.entities.properties.rulers.euclidean import EuclideanRuler
from hoopgn.entities.properties.normalizers.normalizer import PropertyNormalizer
from hoopgn.entities.properties.evaluators.area import AreaEvaluator
from hoopgn.entities.properties.normalizers.area import AreaNormalizer

from hoopgn.entities.properties.property import Property


@dataclass(kw_only=True)
class AreaPropertyConfig(Property.Config):
    ruler: PropertyRuler.Config = EuclideanRuler.Config()
    encoder: PropertyEncoder.Config = AreaEncoderConfig()
    normalizer: PropertyNormalizer.Config = AreaNormalizer.Config()
    evaluator: PropertyEvaluator.Config = AreaEvaluator.Config(
        area=CalvinAreaConfig(),
    )
    parameter: PropertyParameter.Config = EuclideanParameter.Config()

    extractor: PropertyExtractor.Config = CGTAreaExtractor.Config(
        field_name="AreaEuler"
    )
