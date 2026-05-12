from dataclasses import dataclass
from typing import TypeVar, Type, cast

from heca.classes.config import Configurable, ConfigurableMeta

S = TypeVar("S", bound="Registerable")


class RegisterableMeta(ConfigurableMeta):
    query_class_registry: dict[type, type] = {}

    def __init__(cls, name, bases, namespaces):
        super().__init__(name, bases, namespaces)
        query_cls = namespaces.get("Query")
        if query_cls is not None and name != "Registerable":
            RegisterableMeta.query_class_registry[query_cls] = cls

    @classmethod
    def from_query(cls, query):
        query_type = type(query)

        target_cls = RegisterableMeta.query_class_registry.get(query_type)

        if target_cls is None:
            raise ValueError()

        return target_cls


class Registerable(Configurable, metaclass=RegisterableMeta):
    registry: dict["Registerable.Query", "Registerable"] = {}

    @dataclass(frozen=True, kw_only=True)
    class Query:
        label: str

    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        pass

    @classmethod
    def search(cls: Type[S], query: "Registerable.Query") -> S:
        target_cls = RegisterableMeta.from_query(query)

        if query not in cls.registry:
            cls.registry[query] = cast(S, target_cls(target_cls.Config()))
        return cast(S, cls.registry[query])
