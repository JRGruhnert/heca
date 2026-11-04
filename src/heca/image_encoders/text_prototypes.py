import numpy as np
import torch

from dataclasses import dataclass, field

from heca.data.data import TDImage
from heca.data.reference import Keypoint
from heca.image_encoders.image_encoder import ImageEncoder, device_from_spec
from heca.misc import hardware, logger
from heca.scenes.scene import Scene


class TextPrototypeEncoder(ImageEncoder):
    """Which state each entity is in, scored by comparing the frame to prompts.

    One entity's answer set is turned into text embeddings, the frame into an
    image embedding, and the answer with the highest cosine wins -- so a state is
    classified by a model that was never trained on this scene, and "how sure" is
    the softmax over those cosines.
    """

    @dataclass(kw_only=True)
    class Config(ImageEncoder.Config):
        # per answer, {phrase} is the entity's noun phrase and {answer} its state
        templates: list[str] = field(
            default_factory=lambda: [
                "a photo of the {phrase} that is {answer}",
                "the {phrase} is {answer}",
            ]
        )
        device: str = ""
        # cosine scale before the softmax: the answer scores differ by ~1e-2
        logit_scale: float = 100.0

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = device_from_spec(cfg.device) or hardware.device
        self.model = None
        self.entity_labels: list[str] = []
        self.prototypes: dict[str, torch.Tensor] = {}

    def load(self) -> None:
        """Bring up the model, once: the prompt build and the reads share it."""
        raise NotImplementedError

    def embed_text(self, texts: list[str]) -> torch.Tensor:
        """(len(texts), D) unit vectors."""
        raise NotImplementedError

    def embed_image(self, rgb: np.ndarray) -> torch.Tensor:
        """(D,) unit vector for one (H, W, 3) uint8 frame."""
        raise NotImplementedError

    @staticmethod
    def as_uint8(image: TDImage) -> np.ndarray:
        rgb = image.rgb
        if isinstance(rgb, torch.Tensor):
            rgb = (rgb.detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        return np.asarray(rgb, dtype=np.uint8)

    @staticmethod
    def phrase(label: str, entity) -> str:
        return entity.cfg.phrase or "".join(c for c in label if not c.isdigit())

    def prepare_for_scene(self, cfg: Scene.Config):
        scene = Scene.get(cfg)
        wanted = {
            label: entity
            for label, entity in scene.entities.items()
            if entity.cfg.n_states > 1 and entity.cfg.answers
        }
        if not wanted:
            return
        self.load()
        self.prototypes = {}
        for label, entity in wanted.items():
            phrase = self.phrase(label, entity)
            answers = [f"The {phrase} is {answer}." for answer in entity.cfg.answers]
            embeddings = self.embed_text(answers)
            self.prototypes[label] = torch.nn.functional.normalize(embeddings, dim=-1)
        self.entity_labels = list(wanted)
        logger.info(
            f"{type(self).__name__}: {len(self.entity_labels)} entities with states"
            f" ({', '.join(self.entity_labels)}) on {self.device}"
        )

    def reset_tracking(self) -> None:
        return None

    def extract_poses(self, image: TDImage) -> dict[str, Keypoint]:
        return {}

    def extract_states(self, image: TDImage) -> dict[str, tuple[int, float]]:
        """The most likely answer of every entity that has more than one."""
        if not self.prototypes:
            return {}
        self.load()
        image_feature = self.embed_image(self.as_uint8(image))
        states: dict[str, tuple[int, float]] = {}
        for label, answers in self.prototypes.items():
            logits = answers @ image_feature * self.cfg.logit_scale
            probs = torch.softmax(logits, dim=0).detach().cpu().numpy()
            state = int(probs.argmax())
            states[label] = (state, float(probs[state]))
        return states
