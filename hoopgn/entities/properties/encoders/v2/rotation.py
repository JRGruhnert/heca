from dataclasses import dataclass

from hoopgn.entities.properties.encoders.encoder import PropertyEncoder


@dataclass(kw_only=True)
class QuaternionEncoderConfig(PropertyEncoder.Config):
    query: PropertyEncoder.Query = PropertyEncoder.Query(
        label="Quat",
    )
    in_dim: int = 4
