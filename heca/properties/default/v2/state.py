from dataclasses import dataclass

from heca.properties.encoders.encoder import PropertyEncoder
from heca.properties.encoders.v2.state import StateEncoder
from heca.properties.evaluators.domain import DomainEvaluator
from heca.properties.evaluators.evaluator import PropertyEvaluator
from heca.properties.extractors.gt import CGTExtractor
from heca.properties.extractors.extractor import PropertyExtractor
from heca.properties.normalizers.domain import DomainNormalizer
from heca.properties.normalizers.normalizer import PropertyNormalizer
from heca.agents.scenes.parameters.domain import DomainParameter
from heca.agents.scenes.parameters.parameter import PropertyParameter
from heca.properties.rulers.domain import DomainRuler
from heca.properties.rulers.ruler import PropertyRuler
from heca.properties.property import Property


class StateProperty(Property):
    @dataclass(kw_only=True)
    class Config(Property.Config):
        ruler: PropertyRuler.Config = DomainRuler.Config()
        encoder: PropertyEncoder.Config = StateEncoder.Config(
            query=StateEncoder.Query(),
        )
        normalizer: PropertyNormalizer.Config = DomainNormalizer.Config()
        evaluator: PropertyEvaluator.Config = DomainEvaluator.Config()
        parameter: PropertyParameter.Config = DomainParameter.Config()
