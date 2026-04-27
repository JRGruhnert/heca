from dataclasses import dataclass, field

from heca.properties.property import Property
from heca.properties.rulers.ruler import PropertyRuler
from heca.properties.encoders.encoder import PropertyEncoder
from heca.properties.extractors.extractor import PropertyExtractor
from heca.properties.evaluators.evaluator import PropertyEvaluator
from heca.properties.parameters.parameter import PropertyParameter
from heca.properties.extractors.gt import CGTExtractor
from heca.properties.normalizers.normalizer import PropertyNormalizer


class PropertyV1(Property):
    @dataclass(kw_only=True)
    class Config(Property.Config):
        ruler: PropertyRuler.Config
        encoder: PropertyEncoder.Config
        parameter: PropertyParameter.Config
        evaluator: PropertyEvaluator.Config
        normalizer: PropertyNormalizer.Config
        extractor: PropertyExtractor.Config = field(init=False)

        def __post_init__(self):
            self.extractor = CGTExtractor.Config(field_name=self.label)
