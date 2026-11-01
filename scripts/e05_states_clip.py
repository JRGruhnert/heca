"""Sample the CLIP state readout (CPU-capable)."""

from heca.image_encoders.clip_encoder import ClipEncoder
from scripts.common.states import register, sample

register("clip", ClipEncoder.Config)

if __name__ == "__main__":
    sample("clip")
