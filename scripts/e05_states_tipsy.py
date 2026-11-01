"""Sample the TiPS v2 state readout (CPU-capable)."""

from heca.image_encoders.tipsy_encoder import TipsyEncoder
from scripts.common.states import register, sample

register("tipsy", TipsyEncoder.Config)

if __name__ == "__main__":
    sample("tipsy")
