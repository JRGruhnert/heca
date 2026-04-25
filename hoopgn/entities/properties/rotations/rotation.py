from dataclasses import dataclass

from hoopgn.entities.features.extractors.extractor import PropertyExtractor
from hoopgn.entities.properties.encoders.encoder import PropertyEncoder
from hoopgn.properties.features.evaluators.evaluator import (
    PropertyEvaluator,
)

from hoopgn.properties.features.conditions.condition import (
    PropertyCondition,
)

from hoopgn.properties.features.normalizers.normalizer import PropertyNormalizer
from hoopgn.properties.features.rulers.ruler import PropertyRuler
from hoopgn.entities.properties.property import Property


@dataclass
class RotationConfig(Property.Config):
    ruler: PropertyRuler.Config
    encoder: PropertyEncoder.Config
    evaluator: PropertyEvaluator.Config
    condition: PropertyCondition.Config
    extractor: PropertyExtractor.Config
    normalizer: PropertyNormalizer.Config
