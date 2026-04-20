from abc import ABC, ABCMeta
from dataclasses import dataclass
from typing import Sequence, TypeVar, Type, cast

T = TypeVar("T", bound="ConfigurableClass")
S = TypeVar("S", bound="RegisterableClass")


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
    def from_config(cls: Type[T], cfg: "ConfigurableClass.Config") -> T:
        return cast(T, ConfigurableMeta.from_config(cfg))


class RegisterableClass(ConfigurableClass):
    _registry: dict["RegisterableClass.Config", "RegisterableClass"] = {}

    @dataclass(kw_only=True)
    class Signature:
        label: str

    @dataclass(kw_only=True)
    class Config(ConfigurableClass.Config):
        signature: "RegisterableClass.Signature"

    @classmethod
    def get(cls: Type[S], signature: "RegisterableClass.Signature") -> S:
        assert isinstance(signature, cls.Signature), f"Must be of type {cls.Signature}"
        assert signature in cls._registry, f"Label '{signature}' not found in registry"
        return cast(S, cls._registry[signature])

    @classmethod
    def load(
        cls: Type[S],
        cfg: "RegisterableClass.Config" | Sequence["RegisterableClass.Config"],
    ):
        if isinstance(cfg, Sequence):
            for single_cfg in cfg:
                cls.load(single_cfg)
        else:

            cls._registry[cfg] = cls.from_config(cfg)

    @classmethod
    def count(cls, label: str | None = None) -> int:
        if label is None:
            return len(cls._registry)
        return sum(1 for sig in cls._registry if sig.signature.label == label)
