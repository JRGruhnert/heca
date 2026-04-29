from abc import abstractmethod
from dataclasses import dataclass
from typing import TypeVar, Type

from heca.classes.register import Registerable

M = TypeVar("M", bound="Mergeable")


class Mergeable(Registerable):
    @dataclass(frozen=True, kw_only=True)
    class Query(Registerable.Query):
        meta: str

    @dataclass(kw_only=True)
    class Config(Registerable.Config):
        pass

    @classmethod
    @abstractmethod
    def merge(cls: Type[M], query: "Mergeable.Query", items: list[M]) -> M:
        raise NotImplementedError()
