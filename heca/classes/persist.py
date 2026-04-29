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
        tag: str = "epoch"
        ending: str

        @classmethod
        def find_matching_files(cls, directory: Path, pattern: str) -> list[Path]:
            return list(directory.glob(pattern))

        @classmethod
        def resolve_path(
            cls, query: "Persistable.Query", epoch: int, tag: str | None = None
        ) -> Path:
            if tag is None:
                tag = cls.tag
            return (
                cls.resolve_directory()
                / Path(query.label)
                / f"ckpt_{tag}_{epoch}{cls.ending}"
            )

        @classmethod
        def get_latest(cls, query: "Persistable.Query") -> Path | None:
            directory = cls.resolve_directory() / Path(query.label)
            pattern = f"ckpt_{cls.tag}_*.pt"
            matching_files = cls.find_matching_files(directory, pattern)
            epochs = []
            for file in matching_files:
                try:
                    epoch_str = file.stem.split("_")[-1]
                    epoch = int(epoch_str)
                    epochs.append(epoch)
                except (IndexError, ValueError):
                    continue
            max_epoch = max(epochs) if epochs else None
            if max_epoch is not None:
                return cls.resolve_path(query, tag=cls.tag, epoch=max_epoch)
            else:
                return None

        @classmethod
        @abstractmethod
        def resolve_directory(cls) -> Path:
            raise NotImplementedError()

    @classmethod
    @abstractmethod
    def load(cls: Type[P], query: "Persistable.Query") -> P:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def save(cls: Type[P], query: "Persistable.Query", epoch: int, tag: str) -> bool:
        raise NotImplementedError()
