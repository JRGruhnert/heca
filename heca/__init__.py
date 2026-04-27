from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar, Type, cast

C = TypeVar("C", bound="Configurable")
P = TypeVar("P", bound="Persistable")
S = TypeVar("S", bound="Searchable")


class ConfigurableMeta(ABCMeta):
    cfg_class_registry: dict[type, type] = {}

    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        if cls.__name__ != "Configurable":
            config = attrs.get("Config")
            assert config is not None
            ConfigurableMeta.cfg_class_registry[config] = cls

    @staticmethod
    def from_config(config: "Configurable.Config") -> "Configurable":
        subclass = ConfigurableMeta.cfg_class_registry.get(type(config))
        assert subclass is not None
        return subclass(config)


class SearchableMeta(ConfigurableMeta):
    query_class_registry: dict[type, type] = {}

    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        if cls.__name__ != "Searchable":
            query = attrs.get("Query")
            assert query is not None
            SearchableMeta.query_class_registry[query] = cls

    @staticmethod
    def from_query(query: "Searchable.Query") -> "Searchable":
        query_subclass = SearchableMeta.query_class_registry.get(type(query))
        assert query_subclass is not None
        for cfg_class, cfg_subclass in SearchableMeta.cfg_class_registry.items():
            if isinstance(query_subclass, cfg_subclass):
                return cast(Searchable, ConfigurableMeta.from_config(cfg_class()))
        raise ValueError()


class PersistableMeta(SearchableMeta):
    disk_class_registry: dict[type, type] = {}

    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        if cls.__name__ != "Persistable":
            disk = attrs.get("Disk")
            assert disk is not None
            PersistableMeta.disk_class_registry[disk] = cls

    @staticmethod
    def from_disk(disk: "Persistable.Disk") -> "Persistable":
        disk_subclass = PersistableMeta.disk_class_registry.get(type(disk))
        assert disk_subclass is not None
        for query_class, query_subclass in PersistableMeta.query_class_registry.items():
            if isinstance(disk_subclass, query_subclass):
                return cast(Persistable, SearchableMeta.from_query(query_class()))
        raise ValueError()


class Configurable(ABC, metaclass=ConfigurableMeta):
    @dataclass(kw_only=True)
    class Config:
        pass

    @classmethod
    def create(cls: Type[C], cfg: "Configurable.Config") -> C:
        return cast(C, ConfigurableMeta.from_config(cfg))


class Searchable(Configurable, metaclass=SearchableMeta):
    registry: dict["Searchable.Query", "Searchable"] = {}

    @dataclass(frozen=True, kw_only=True)
    class Query(Configurable):
        label: str

    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        pass

    @classmethod
    def search(cls: Type[S], query: "Searchable.Query") -> S:
        if query not in cls.registry:
            cls.registry[query] = cast(S, SearchableMeta.from_query(query))
        assert query in cls.registry
        return cast(S, cls.registry[query])


class Persistable(Searchable, metaclass=PersistableMeta):
    @dataclass(frozen=True, kw_only=True)
    class Query(Searchable.Query):
        pass

    @dataclass(kw_only=True)
    class Config(Searchable.Config):
        pass

    @dataclass(kw_only=True)
    class Disk(Searchable):
        root: Path = Path("data")
        ending: str = ".pt"

        @classmethod
        @abstractmethod
        def resolve_path(cls, query: Searchable.Query) -> Path:
            raise NotImplementedError()

    @abstractmethod
    def load(self, query: "Persistable.Query"):
        raise NotImplementedError()

    @abstractmethod
    def save(self, query: "Persistable.Query") -> None:
        raise NotImplementedError()
