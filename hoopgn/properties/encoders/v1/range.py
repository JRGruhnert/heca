from dataclasses import dataclass

from hoopgn.properties.encoders.encoder import PropertyEncoder


@dataclass(kw_only=True)
class RangeEncoderConfig(PropertyEncoder.Config):
    query: PropertyEncoder.Query = PropertyEncoder.Query(
        label="Range",
    )
    in_dim: int = 1
    hidden_dim: int = 8
