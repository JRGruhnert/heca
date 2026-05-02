from dataclasses import dataclass
import copy
from types import MappingProxyType

from heca.classes.config import Configurable
from heca.classes.merge import Mergeable
from heca.entities.entity import Entity
from heca.misc.td import TDEntity, TDScene
from heca.properties.rulers.ruler import PropertyRuler
from heca.properties.encoders.encoder import PropertyEncoder
from heca.properties.extractors.extractor import PropertyExtractor
from heca.properties.evaluators.evaluator import PropertyEvaluator
from heca.properties.aggregators.aggregator import PropertyAggregator
from heca.properties.normalizers.normalizer import PropertyNormalizer


class Precon(Mergeable):
    @dataclass(kw_only=True)
    class Config(Mergeable.Config):
        pass

    def __init__(self, values: dict[Entity.Query, TDEntity]):
        self._values = MappingProxyType(values)

    @property
    def values(self) -> list[TDEntity]:
        return list(self._values.values())

    def __getitem__(self, query: Entity.Query) -> TDEntity:
        return self._values[query]


# TODO: make classs for precon and postcon that can merge and cluster from other precons
