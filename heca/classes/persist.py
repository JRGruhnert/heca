from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Type, TypeVar, cast

from heca.classes.register import Registerable, RegisterableMeta

P = TypeVar("P", bound="Persistable")


class PersistableMeta(RegisterableMeta):
    disk_class_registry: dict[type, type] = {}

    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        if cls.__name__ != "Persistable":
            disk = attrs.get("Disk")
            assert disk is not None
            PersistableMeta.disk_class_registry[disk] = cls

    @staticmethod
    def from_disk(disk: "Persistable.File") -> "Persistable":
        disk_subclass = PersistableMeta.disk_class_registry.get(type(disk))
        assert disk_subclass is not None
        for query_class, query_subclass in PersistableMeta.query_class_registry.items():
            if isinstance(disk_subclass, query_subclass):
                return cast(Persistable, RegisterableMeta.from_query(query_class()))
        raise ValueError()


class Persistable(Registerable, metaclass=PersistableMeta):
    @dataclass(frozen=True, kw_only=True)
    class Query(Registerable.Query):
        pass

    @dataclass(kw_only=True)
    class Config(Registerable.Config):
        pass

    @dataclass(frozen=True, kw_only=True)
    class File(Registerable):
        root: Path = Path("data")
        folder: str
        ending: str

        @classmethod
        def resolve_directory(cls, query: "Persistable.Query") -> Path:
            path = cls.root / Path(cls.folder) / Path(query.label)
            path.mkdir(parents=True, exist_ok=True)
            return path

    @classmethod
    @abstractmethod
    def load(cls: Type[P], query: "Persistable.Query") -> P:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def save(cls: Type[P], query: "Persistable.Query") -> bool:
        raise NotImplementedError()
