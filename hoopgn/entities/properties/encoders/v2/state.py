from dataclasses import dataclass

from hoopgn.entities.properties.encoders.encoder import PropertyEncoder


@dataclass(kw_only=True)
class StateEncoderConfig(PropertyEncoder.Config):
    query: PropertyEncoder.Query = PropertyEncoder.Query(
        label="State",
    )
    in_dim: int = 1
    hidden_dim: int = 8
