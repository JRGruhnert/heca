from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar, Type, cast

T = TypeVar("T", bound="ConfigurableClass")
S = TypeVar("S", bound="SearchableClass")
V = TypeVar("V", bound="StoragableClass")


class ConfigurableMeta(ABCMeta):
    config_class_registry: dict[type, type] = {}

    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        if cls.__name__ != "ConfigurableClass":
            # Ensure each subclass defines its own Config class
            config_cls = attrs.get("Config", None)
            assert config_cls is not None, f"{cls.__name__} must define a Config class"
            ConfigurableMeta.config_class_registry[config_cls] = cls

    @staticmethod
    def from_config(cfg: "ConfigurableClass.Config") -> "ConfigurableClass":
        subclass = ConfigurableMeta.config_class_registry.get(type(cfg))
        assert subclass is not None
        return subclass(cfg)


class ConfigurableClass(ABC, metaclass=ConfigurableMeta):
    @dataclass(kw_only=True)
    class Config:
        pass

    @classmethod
    def from_config(cls: Type[T], cfg: "ConfigurableClass.Config") -> T:
        return cast(T, ConfigurableMeta.from_config(cfg))

    @classmethod
    def from_configs(cls: Type[T], cfgs: list["ConfigurableClass.Config"]) -> list[T]:
        return [cls.from_config(cfg) for cfg in cfgs]


class SearchableClass(ConfigurableClass):
    registry: dict["SearchableClass.Query", "SearchableClass"] = {}

    @dataclass(kw_only=True)
    class Query(ConfigurableClass.Config):
        label: str

        def __eq__(self, other):
            if not isinstance(other, SearchableClass.Query):
                return NotImplemented
            return self.label == other.label

    @dataclass(kw_only=True)
    class Config(ConfigurableClass.Config):
        query: "SearchableClass.Query"

    @classmethod
    def search(cls: Type[S], query: "SearchableClass.Query") -> S:
        if query not in cls.registry:
            choice = None
            for cfg in cls.config_class_registry:
                if isinstance(cfg, "SearchableClass.Config"):
                    if cfg.query == query:
                        choice = cfg

            assert choice is not None
            cls.registry[query] = cls.from_config(choice)
        assert query in cls.registry
        return cast(S, cls.registry[query])


class StoragableClass(SearchableClass):
    @dataclass(kw_only=True)
    class Query(SearchableClass.Query):
        root: Path = Path("data")
        ending: str = ".pt"

    @dataclass(kw_only=True)
    class Config(SearchableClass.Config):
        pass

    @classmethod
    def from_disk(cls: Type[V], cfg: "StoragableClass.Config") -> V:
        instance = cls.from_config(cfg)
        if not isinstance(cfg.query, StoragableClass.Query):
            raise TypeError("cfg.query must be a StoragableClass.Query")
        path = cls.resolve_path(cfg.query)
        instance.load(path, cfg.query.label)
        return instance

    @classmethod
    @abstractmethod
    def resolve_path(cls: Type[V], query: "StoragableClass.Query") -> Path:
        raise NotImplementedError()

    @abstractmethod
    def load(self, path: Path, label: str) -> None:
        raise NotImplementedError()

    @abstractmethod
    def save(self, path: Path, label: str) -> None:
        raise NotImplementedError()
