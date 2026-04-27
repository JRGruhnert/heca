from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
import torch
from heca.properties.extractors.extractor import PropertyExtractor
from heca.misc.classes import Persistable


class ImagePropertyExtractor(PropertyExtractor, Persistable):
    @dataclass(kw_only=True)
    class Query(Persistable.Query):
        scene: str
        entity: str
        property: str

    @dataclass(kw_only=True)
    class Config(PropertyExtractor.Config, Persistable.Config):
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
            / Path(query.scene)
            / Path(query.entity)
            / Path(query.property)
            / f"{query.label}.{query.ending}"
        )

    def load(self, query: "ImagePropertyExtractor.Query") -> None:
        raise NotImplementedError()

    def save(self, query: "ImagePropertyExtractor.Query") -> None:
        raise NotImplementedError()
