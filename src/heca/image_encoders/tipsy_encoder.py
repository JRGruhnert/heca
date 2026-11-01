import numpy as np
import torch
import torch.nn.functional as functional

from dataclasses import dataclass

from heca.image_encoders.text_prototypes import TextPrototypeEncoder
from heca.misc import logger


class TipsyEncoder(TextPrototypeEncoder):
    """State readout through TiPS v2 (google/tipsv2-b14)."""

    @dataclass(kw_only=True)
    class Config(TextPrototypeEncoder.Config):
        tag: str = "google/tipsv2-b14"
        side: int = 448  # the notebook's input size for this model
        backend: str = "tipsy"

    def load(self) -> None:
        if self.model is not None:
            return
        from transformers import AutoModel

        self.model = (
            AutoModel.from_pretrained(self.cfg.tag, trust_remote_code=True)
            .to(self.device)
            .eval()
        )
        logger.info(f"{type(self).__name__}: {self.cfg.tag} on {self.device}")

    def embed_text(self, texts: list[str]) -> torch.Tensor:
        with torch.inference_mode():
            features = self.model.encode_text(texts)  # type: ignore[union-attr]
        return functional.normalize(features, dim=-1)

    def embed_image(self, rgb: np.ndarray) -> torch.Tensor:
        side = self.cfg.side
        pixels = torch.from_numpy(np.asarray(rgb)).permute(2, 0, 1)[None]
        pixels = functional.interpolate(
            pixels.float() / 255.0, size=(side, side), mode="bicubic", align_corners=False
        )
        with torch.inference_mode():
            out = self.model.encode_image(pixels.to(self.device))  # type: ignore[union-attr]
        return functional.normalize(out.cls_token.reshape(-1), dim=-1)
