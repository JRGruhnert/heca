from dataclasses import dataclass

from hoopgn.entities.properties.encoders.encoder import PropertyEncoder


@dataclass(kw_only=True)
class AreaEncoderConfig(PropertyEncoder.Config):
    query: PropertyEncoder.Query = PropertyEncoder.Query(
        label="AreaEuler",
    )
    in_dim: int = 6
