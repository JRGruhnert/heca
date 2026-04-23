from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Sequence, TypeVar, Type, cast

T = TypeVar("T", bound="ConfigurableClass")
S = TypeVar("S", bound="RegisterableClass")
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


class RegisterableClass(ConfigurableClass):
    registry: dict["RegisterableClass.Config", "RegisterableClass"] = {}

    @dataclass(kw_only=True)
    class Signature:
        label: str

    @dataclass(kw_only=True)
    class Config(ConfigurableClass.Config):
        sig: "RegisterableClass.Signature"

    @classmethod
    def get(cls: Type[S], signature: "RegisterableClass.Signature") -> S:
        assert isinstance(signature, cls.Signature), f"Must be of type {cls.Signature}"
        assert signature in cls.registry, f"Label '{signature}' not found in registry"
        return cast(S, cls.registry[signature])

    @classmethod
    def load(
        cls: Type[S],
        cfg: "RegisterableClass.Config" | Sequence["RegisterableClass.Config"],
    ):
        if isinstance(cfg, Sequence):
            for single_cfg in cfg:
                cls.load(single_cfg)
        else:
            cls.registry[cfg] = cls.from_config(cfg)


class StoragableClass(RegisterableClass):
    registry: dict["StoragableClass.Config", "StoragableClass"] = {}

    @dataclass(kw_only=True)
    class Signature(RegisterableClass.Signature):
        label: str

    @dataclass(kw_only=True)
    class Config(RegisterableClass.Config):
        sig: "StoragableClass.Signature"

    @classmethod
    def get(cls: Type[V], sig: "StoragableClass.Signature") -> V:
        assert isinstance(sig, cls.Signature), f"Must be of type {cls.Signature}"
        assert sig in cls.registry, f"Label '{sig}' not found in registry"
        return cast(V, cls.registry[sig])

    @classmethod
    def load(
        cls: Type[V],
        cfg: "StoragableClass.Config" | Sequence["StoragableClass.Config"],
    ):
        if isinstance(cfg, Sequence):
            for c in cfg:
                cls.load(c)
        else:
            cls.registry[cfg] = cls.from_config(cfg)

    @classmethod
    def path(cls, sig: "StoragableClass.Signature") -> Path:
        raise NotImplementedError()

    @classmethod
    def from_disk(cls: Type[V], path: str) -> V:
        # Implement loading from disk logic here
        raise NotImplementedError()

    @classmethod
    def to_disk(cls, path: str):
        # Implement saving to disk logic here
        raise NotImplementedError()

    @property
    def storage_path(self) -> Path:
        if isinstance(self.parent, StoragableClass):
            return self.parent.storage_path / self.storage_name
        else:
            return Path(self.storage_name)
