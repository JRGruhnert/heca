from dataclasses import dataclass

from heca.properties.encoders.encoder import PropertyEncoder
from heca.properties.encoders.v2.domain import DomainEncoder
from heca.properties.evaluators.domain import DomainEvaluator
from heca.properties.evaluators.evaluator import PropertyEvaluator
from heca.properties.rulers.domain import DomainRuler
from heca.properties.rulers.ruler import PropertyRuler
from heca.properties.property import Property


class DomainProperty(Property):
    @dataclass(kw_only=True)
    class Config(Property.Config):
        ruler: PropertyRuler.Config = DomainRuler.Config()
        encoder: PropertyEncoder.Query = DomainEncoder.Query()
        evaluator: PropertyEvaluator.Config = DomainEvaluator.Config()
