from dataclasses import dataclass
from typing import TypeVar, Type, cast

from heca.classes.config import Configurable, ConfigurableMeta

S = TypeVar("S", bound="Searchable")


class SearchableMeta(ConfigurableMeta):
    query_class_registry: dict[type, type] = {}

    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        if cls.__name__ != "Searchable":
            query = attrs.get("Query")
            assert query is not None
            SearchableMeta.query_class_registry[query] = cls

    @staticmethod
    def from_query(query: "Registerable.Query") -> "Registerable":
        query_subclass = SearchableMeta.query_class_registry.get(type(query))
        assert query_subclass is not None
        for cfg_class, cfg_subclass in SearchableMeta.cfg_class_registry.items():
            if isinstance(query_subclass, cfg_subclass):
                return cast(Registerable, ConfigurableMeta.from_config(cfg_class()))
        raise ValueError()


class Registerable(Configurable, metaclass=SearchableMeta):
    registry: dict["Registerable.Query", "Registerable"] = {}

    @dataclass(frozen=True, kw_only=True)
    class Query(Configurable):
        label: str

    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        pass

    @classmethod
    def search(cls: Type[S], query: "Registerable.Query") -> S:
        if query not in cls.registry:
            cls.registry[query] = cast(S, SearchableMeta.from_query(query))
        assert query in cls.registry
        return cast(S, cls.registry[query])
