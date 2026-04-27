from abc import ABC, ABCMeta
from dataclasses import dataclass
from typing import TypeVar, Type, cast

C = TypeVar("C", bound="Configurable")


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


class Configurable(ABC, metaclass=ConfigurableMeta):
    @dataclass(kw_only=True)
    class Config:
        pass

    @classmethod
    def create(cls: Type[C], cfg: "Configurable.Config") -> C:
        return cast(C, ConfigurableMeta.from_config(cfg))
