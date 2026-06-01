import abc
import torch

from dataclasses import dataclass

from heca.classes.register import Registerable
from heca.entities.entity import Entity
from heca.misc.td import TDImage


class MolmoExtractor(Registerable):
    @dataclass(kw_only=True)
    class Config(Registerable.Config):
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
