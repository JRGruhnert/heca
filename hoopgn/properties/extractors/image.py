from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
import torch
from hoopgn.properties.extractors.extractor import PropertyExtractor
from hoopgn.properties.property import Property
from hoopgn.misc.classes import StoragableClass


class ImagePropertyExtractor(PropertyExtractor, StoragableClass):
    @dataclass(kw_only=True)
    class Query(StoragableClass.Query):
        property: Property.Query

    @dataclass(kw_only=True)
    class Config(PropertyExtractor.Config, StoragableClass.Config):
        pass

    def __init__(self, cfg: Config):
        self.cfg = cfg

    @abstractmethod
    def __call__(self, observation) -> torch.Tensor:
        raise NotImplementedError()

    @classmethod
    def resolve_path(cls, query: "ImagePropertyExtractor.Query") -> Path:
        return (
            Path(query.root)
            / Path(query.property.entity.env.label)
            / Path(query.property.entity.label)
            / Path(query.property.label)
            / f"{query.label}.pt"
        )

    def load(self, path: Path, label: str) -> None:
        raise NotImplementedError()

    def save(self, path: Path, label: str) -> None:
        raise NotImplementedError()
