from abc import ABC, ABCMeta
from dataclasses import dataclass
from typing import TypeVar, Type, cast

C = TypeVar("C", bound="Configurable")


class ConfigurableMeta(ABCMeta):
    cfg_class_registry: dict[type, type] = {}

    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        config = attrs.get("Config")
        if config is not None and name != "Configurable":
            ConfigurableMeta.cfg_class_registry[config] = cls

    @staticmethod
    def from_config(config):
        for cfg_cls, subclass in ConfigurableMeta.cfg_class_registry.items():
            if isinstance(config, cfg_cls):
                return subclass(config)
        raise ValueError()


class Configurable(ABC, metaclass=ConfigurableMeta):
    @dataclass(kw_only=True)
    class Config:
        pass

    @classmethod
    def create(cls: Type[C], cfg: "Configurable.Config") -> C:
        return cast(C, ConfigurableMeta.from_config(cfg))
