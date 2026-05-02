from dataclasses import dataclass

from heca.properties.encoders.encoder import PropertyEncoder
from heca.properties.encoders.v2.state import StateEncoder
from heca.properties.evaluators.state import StateEvaluator
from heca.properties.evaluators.evaluator import PropertyEvaluator
from heca.properties.rulers.state import StateRuler
from heca.properties.rulers.ruler import PropertyRuler
from heca.properties.property import Property


class StateProperty(Property):
    @dataclass(kw_only=True)
    class Config(Property.Config):
        ruler: PropertyRuler.Config = StateRuler.Config()
        encoder: PropertyEncoder.Query = StateEncoder.Query()
        evaluator: PropertyEvaluator.Config = StateEvaluator.Config()
