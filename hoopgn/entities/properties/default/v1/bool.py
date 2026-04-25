from dataclasses import dataclass

from hoopgn.entities.properties.encoders.encoder import PropertyEncoder
from hoopgn.entities.properties.encoders.v1.bool import BoolEncoderConfig
from hoopgn.entities.properties.evaluators.evaluator import PropertyEvaluator
from hoopgn.entities.properties.evaluators.threshold import (
    ThresholdEvaluator,
)

from hoopgn.entities.properties.extractors.c_gt import (
    CGTExtractor,
)
from hoopgn.entities.properties.extractors.extractor import PropertyExtractor
from hoopgn.entities.properties.normalizers.ignore import IgnoreNormalizer
from hoopgn.entities.properties.normalizers.normalizer import PropertyNormalizer
from hoopgn.entities.properties.parameters.binary import (
    BinaryParameter,
)
from hoopgn.entities.properties.parameters.parameter import PropertyParameter
from hoopgn.entities.properties.rulers.binary import (
    BinaryRuler,
)

from hoopgn.entities.properties.rulers.ruler import PropertyRuler
from hoopgn.entities.properties.property import Property


@dataclass(kw_only=True)
class BoolPropertyConfig(Property.Config):
    ruler: PropertyRuler.Config = BinaryRuler.Config()
    encoder: PropertyEncoder.Config = BoolEncoderConfig()
    normalizer: PropertyNormalizer.Config = IgnoreNormalizer.Config()
    evaluator: PropertyEvaluator.Config = ThresholdEvaluator.Config()
    parameter: PropertyParameter.Config = BinaryParameter.Config()
    extractor: PropertyExtractor.Config = CGTExtractor.Config(field_name="Bool")
