from abc import ABC, ABCMeta
from dataclasses import dataclass
from typing import TypeVar, Type, cast

T = TypeVar("T", bound="ConfigurableClass")


class ConfigurableMeta(ABCMeta):
    registry: dict[type, type] = {}

    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        if cls.__name__ != "ConfigurableClass":
            # Ensure each subclass defines its own Config class
            config_cls = attrs.get("Config", None)
            assert config_cls is not None, f"{cls.__name__} must define a Config class"
            ConfigurableMeta.registry[config_cls] = cls

    @staticmethod
    def from_config(cfg: "ConfigurableClass.Config") -> "ConfigurableClass":
        subclass = ConfigurableMeta.registry.get(type(cfg))
        assert subclass is not None
        return subclass(cfg)


class ConfigurableClass(ABC, metaclass=ConfigurableMeta):
    @dataclass(kw_only=True)
    class Config:
        pass

    @classmethod
    def from_config(cls: Type[T], cfg: Config) -> T:
        return cast(T, ConfigurableMeta.from_config(cfg))
