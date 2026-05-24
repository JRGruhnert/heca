from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Type, TypeVar, cast

from heca.classes.register import Registerable, RegisterableMeta
from heca.misc import logger

P = TypeVar("P", bound="Persistable")


class PersistableMeta(RegisterableMeta):
    location_class_registry: dict[type, type] = {}

    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        if cls.__name__ != "Persistable":
            location = getattr(cls, "Location", None)
            assert location is not None
            PersistableMeta.location_class_registry[location] = cls

    @staticmethod
    def from_disk(disk: "Persistable.Location") -> "Persistable":
        disk_subclass = PersistableMeta.location_class_registry.get(type(disk))
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
    class Location:
        root: Path = Path("data")
        folder: str

        @classmethod
        def resolve(cls, query: "Persistable.Query") -> Path:
            path = cls.root / Path(cls.folder) / Path(query.label)
            path.mkdir(parents=True, exist_ok=True)
            return path

    @classmethod
    def load(
        cls: Type[P],
        query: "Persistable.Query",
        location: "Persistable.Location",
    ) -> P:
        path = location.resolve(query)
        persistable = cls.search(query)
        logger.info(f"Loading {type(persistable)} {query.label} data from {path}")
        persistable._load(path)
        return persistable

    @classmethod
    def save(
        cls: Type[P],
        query: "Persistable.Query",
        location: "Persistable.Location",
    ) -> bool:
        path = location.resolve(query)
        persistable = cls.search(query)
        logger.info(f"Saving {type(persistable)} {query.label} data to {path}")
        persistable._save(path)
        return True

    @abstractmethod
    def _load(self, path: Path):
        raise NotImplementedError()

    @abstractmethod
    def _save(self, path: Path):
        raise NotImplementedError()
