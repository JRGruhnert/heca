import numpy as np
import torch

from dataclasses import dataclass

from heca.image_encoders.text_prototypes import TextPrototypeEncoder
from heca.misc import logger


def _load(model_class, tag: str):
    """Load safetensors first: transformers refuses .bin files under torch < 2.6."""
    try:
        return model_class.from_pretrained(tag, use_safetensors=True)
    except (OSError, ValueError):
        logger.warning(f"{tag}: no safetensors weights, falling back to torch.load")
        return model_class.from_pretrained(tag)


class ClipEncoder(TextPrototypeEncoder):
    """State readout through OpenAI CLIP: text prompts against the frame."""

    @dataclass(kw_only=True)
    class Config(TextPrototypeEncoder.Config):
        tag: str = "openai/clip-vit-base-patch32"
        backend: str = "clip"

    def load(self) -> None:
        if self.model is not None:
            return
        from transformers import CLIPModel, CLIPProcessor

        self.model = _load(CLIPModel, self.cfg.tag).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(self.cfg.tag)
        logger.info(f"{type(self).__name__}: {self.cfg.tag} on {self.device}")

    def embed_text(self, texts: list[str]) -> torch.Tensor:
        with torch.inference_mode():
            feats = self.model.get_text_features(  # type: ignore[union-attr]
                **self.processor(text=texts, return_tensors="pt", padding=True).to(
                    self.device
                )
            )
        return torch.nn.functional.normalize(feats, dim=-1)

    def embed_image(self, rgb: np.ndarray) -> torch.Tensor:
        with torch.inference_mode():
            feats = self.model.get_image_features(  # type: ignore[union-attr]
                **self.processor(images=rgb, return_tensors="pt").to(self.device)
            )
        return torch.nn.functional.normalize(feats, dim=-1)[0]
