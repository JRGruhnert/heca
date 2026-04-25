from dataclasses import dataclass


from hoopgn.environments.calvin import CalvinAreaConfig
from hoopgn.properties.encoders.encoder import PropertyEncoder

from hoopgn.properties.encoders.v1.area import AreaEncoderConfig
from hoopgn.properties.extractors.c_gt_area import CGTAreaExtractor
from hoopgn.properties.extractors.extractor import PropertyExtractor
from hoopgn.properties.parameters.parameter import PropertyParameter
from hoopgn.properties.rulers.ruler import PropertyRuler
from hoopgn.properties.evaluators.evaluator import PropertyEvaluator
from hoopgn.properties.parameters.euclidean import EuclideanParameter
from hoopgn.properties.rulers.euclidean import EuclideanRuler
from hoopgn.properties.normalizers.normalizer import PropertyNormalizer
from hoopgn.properties.evaluators.area import AreaEvaluator
from hoopgn.properties.normalizers.area import AreaNormalizer

from hoopgn.properties.property import Property


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
