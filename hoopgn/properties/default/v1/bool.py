from dataclasses import dataclass

from hoopgn.properties.encoders.encoder import PropertyEncoder
from hoopgn.properties.encoders.v1.bool import BoolEncoderConfig
from hoopgn.properties.evaluators.evaluator import PropertyEvaluator
from hoopgn.properties.evaluators.threshold import (
    ThresholdEvaluator,
)

from hoopgn.properties.extractors.c_gt import (
    CGTExtractor,
)
from hoopgn.properties.extractors.extractor import PropertyExtractor
from hoopgn.properties.normalizers.ignore import IgnoreNormalizer
from hoopgn.properties.normalizers.normalizer import PropertyNormalizer
from hoopgn.properties.parameters.binary import (
    BinaryParameter,
)
from hoopgn.properties.parameters.parameter import PropertyParameter
from hoopgn.properties.rulers.binary import (
    BinaryRuler,
)

from hoopgn.properties.rulers.ruler import PropertyRuler
from hoopgn.properties.property import Property


@dataclass(kw_only=True)
class BoolPropertyConfig(Property.Config):
    ruler: PropertyRuler.Config = BinaryRuler.Config()
    encoder: PropertyEncoder.Config = BoolEncoderConfig()
    normalizer: PropertyNormalizer.Config = IgnoreNormalizer.Config()
    evaluator: PropertyEvaluator.Config = ThresholdEvaluator.Config()
    parameter: PropertyParameter.Config = BinaryParameter.Config()
    extractor: PropertyExtractor.Config = CGTExtractor.Config(field_name="Bool")
