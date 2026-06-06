import torch

from dataclasses import dataclass

from heca.entities.entity import Entity
from heca.environment.scenes.scene import Scene
from heca.image_encoders.image_encoder import ImageEncoder
from heca.misc.td import TDImage


class MolmoEncoder(ImageEncoder):
    @dataclass(kw_only=True)
    class Config(ImageEncoder.Config):
        kp_selection_threshold: float = 0.2
        image_size: tuple[int, int] = (256, 256)

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def extract_entities(
        self,
        image: TDImage,
        entities: list[Entity],
    ) -> torch.Tensor:
        raise NotImplementedError()

    def extract_states(
        self,
        image: TDImage,
        entities: list[Entity],
        kps_raw_2d: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError()

    def prepare_for_scene(self, scene: Scene.Config):
        pass  # By default, no preparation is needed. Override in specific extractors if needed.
