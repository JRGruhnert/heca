from dataclasses import dataclass, field

from heca.properties.aggregators.aggregator import PropertyAggregator
from heca.properties.property import PropertyV1
from heca.properties.rulers.ruler import PropertyRuler
from heca.properties.encoders.encoder import PropertyEncoder
from heca.properties.extractors.extractor import PropertyExtractor
from heca.properties.evaluators.evaluator import PropertyEvaluator
from heca.properties.extractors.gt import CGTExtractor
from heca.properties.normalizers.normalizer import PropertyNormalizer


class PropertyV1(PropertyV1):
    @dataclass(kw_only=True)
    class Config(PropertyV1.Config):
        ruler: PropertyRuler.Config
        encoder: PropertyEncoder.Config
        evaluator: PropertyEvaluator.Config
        normalizer: PropertyNormalizer.Config
        extractor: PropertyExtractor.Config = field(init=False)
        aggregator: PropertyAggregator.Config = field(init=False)

        def __post_init__(self):
            self.extractor = CGTExtractor.Config(field_name=self.label)
            self.aggregator = PropertyAggregator.Config()
