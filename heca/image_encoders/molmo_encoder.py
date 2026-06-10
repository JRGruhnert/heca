import torch

from dataclasses import dataclass
from transformers import AutoProcessor, AutoModelForImageTextToText

from heca.entities.entity import Entity
from heca.environment.scenes.scene import Scene
from heca.image_encoders.image_encoder import ImageEncoder
from heca.misc.td import TDImage


class MolmoEncoder(ImageEncoder):
    @dataclass(kw_only=True)
    class Config(ImageEncoder.Config):
        kp_selection_threshold: float = 0.2
        image_size: tuple[int, int] = (256, 256)
        tag: str = "allenai/Molmo2-4B"

    def __init__(self, cfg: Config):
        self.cfg = cfg

        self.processor = AutoProcessor.from_pretrained(
            self.cfg.tag, trust_remote_code=True
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.cfg.tag,
            trust_remote_code=True,
            dtype=torch.float32,
            device_map="auto",
        )
        self.model.eval()

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
