import abc
import torch

from dataclasses import dataclass

from heca.entities.entity import Entity
from heca.environment.scenes.scene import Scene
from heca.image_extractors.image_extractor import ImageExtractor
from heca.misc.td import TDImage


class MolmoExtractor(ImageExtractor):
    @dataclass(kw_only=True, frozen=True)
    class Query(ImageExtractor.Query):
        label: str = "molmo"

    @dataclass(kw_only=True)
    class Config(ImageExtractor.Config):
        kp_selection_threshold: float = 0.2
        image_size: tuple[int, int] = (256, 256)

    def __init__(self, cfg: Config):
        self.cfg = cfg

    @abc.abstractmethod
    def extract_entities(
        self,
        td_img: TDImage,
        entities: list[Entity],
    ) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def extract_states(
        self,
        td_img: TDImage,
        entities: list[Entity],
        kps_raw_2d: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError()

    def prepare_for_scene(self, scene: Scene.Query):
        pass  # By default, no preparation is needed. Override in specific extractors if needed.
