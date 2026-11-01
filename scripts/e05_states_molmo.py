"""Sample the Molmo state readout (needs the cards: 9.7 GB of weights)."""

from heca.image_encoders.molmo_encoder import MolmoEncoder
from scripts.common.states import register, sample

register("molmo", MolmoEncoder.Config)

if __name__ == "__main__":
    sample("molmo")
