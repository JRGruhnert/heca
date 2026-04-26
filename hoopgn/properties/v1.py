from pathlib import Path
import torch
from dataclasses import dataclass, field
from hoopgn.entities.entity import Entity
from hoopgn.properties.extractors.c_gt_area import CGTAreaExtractor
from hoopgn.properties.property import Property
from hoopgn.entities.v1 import EntityV1
from hoopgn.misc.classes import SearchableClass

from hoopgn.properties.encoders.encoder import PropertyEncoder
from hoopgn.properties.extractors.extractor import (
    PropertyExtractor,
)

from hoopgn.properties.evaluators.evaluator import (
    PropertyEvaluator,
)

from hoopgn.properties.parameters.parameter import PropertyParameter
from hoopgn.properties.rulers.ruler import PropertyRuler

from hoopgn.properties.normalizers.normalizer import (
    PropertyNormalizer,
)


class PropertyV1(Property):
    @dataclass(kw_only=True)
    class Query(Property.Query):
        entity: Entity.Query = EntityV1.Query()

    @dataclass(kw_only=True)
    class Config(Property.Config):
        ruler: PropertyRuler.Config
        encoder: PropertyEncoder.Config
        parameter: PropertyParameter.Config
        evaluator: PropertyEvaluator.Config
        normalizer: PropertyNormalizer.Config
        extractor: PropertyExtractor.Config = field(init=False)

        def __post_init__(self):
            self.extractor = CGTAreaExtractor.Config(field_name=self.query.label)
