from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar, Type, cast

T = TypeVar("T", bound="ConfigClass")
S = TypeVar("S", bound="QueryClass")
V = TypeVar("V", bound="StorageClass")


class ConfigurableMeta(ABCMeta):
    config_class_registry: dict[type, type] = {}

    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        if cls.__name__ != "ConfigClass":
            # Ensure each subclass defines its own Config class
            config_cls = attrs.get("Config", None)
            assert config_cls is not None, f"{cls.__name__} must define a Config class"
            ConfigurableMeta.config_class_registry[config_cls] = cls

    @staticmethod
    def from_config(cfg: "ConfigClass.Config") -> "ConfigClass":
        subclass = ConfigurableMeta.config_class_registry.get(type(cfg))
        assert subclass is not None
        return subclass(cfg)


class ConfigClass(ABC, metaclass=ConfigurableMeta):
    @dataclass(kw_only=True)
    class Config:
        pass

    @classmethod
    def from_config(cls: Type[T], cfg: "ConfigClass.Config") -> T:
        return cast(T, ConfigurableMeta.from_config(cfg))

    @classmethod
    def from_configs(cls: Type[T], cfgs: list["ConfigClass.Config"]) -> list[T]:
        return [cls.from_config(cfg) for cfg in cfgs]


class QueryClass(ConfigClass):
    registry: dict["QueryClass.Query", "QueryClass"] = {}

    @dataclass(kw_only=True)
    class Query(ConfigClass.Config):
        label: str

        def __eq__(self, other):
            if not isinstance(other, QueryClass.Query):
                return NotImplemented
            return self.label == other.label

    @dataclass(kw_only=True)
    class Config(ConfigClass.Config):
        query: "QueryClass.Query"

    @classmethod
    def search(cls: Type[S], query: "QueryClass.Query") -> S:
        if query not in cls.registry:
            choice = None
            for cfg in cls.config_class_registry:
                if isinstance(cfg, "QueryClass.Config"):
                    if cfg.query == query:
                        choice = cfg

            assert choice is not None
            cls.registry[query] = cls.from_config(choice)
        assert query in cls.registry
        return cast(S, cls.registry[query])

    @classmethod
    def search_multiple(cls: Type[S], queries: list["QueryClass.Query"]) -> list[S]:
        return [cls.search(query) for query in queries]


class StorageClass(QueryClass):
    @dataclass(kw_only=True)
    class Query(QueryClass.Query):
        root: Path = Path("data")
        ending: str = ".pt"

    @dataclass(kw_only=True)
    class Config(QueryClass.Config):
        pass

    @classmethod
    def from_disk(cls: Type[V], cfg: "StorageClass.Config") -> V:
        instance = cls.from_config(cfg)
        if not isinstance(cfg.query, StorageClass.Query):
            raise TypeError("cfg.query must be a StorageClass.Query")
        instance.load(cfg.query)
        return instance

    @classmethod
    @abstractmethod
    def resolve_path(cls: Type[V], query: "StorageClass.Query") -> Path:
        raise NotImplementedError()

    @abstractmethod
    def load(self, query: "StorageClass.Query"):
        raise NotImplementedError()

    @abstractmethod
    def save(self, query: "StorageClass.Query") -> None:
        raise NotImplementedError()
