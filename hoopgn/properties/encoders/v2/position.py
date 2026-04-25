from dataclasses import dataclass

from hoopgn.properties.encoders.encoder import PropertyEncoder


@dataclass(kw_only=True)
class PositionEncoderConfig(PropertyEncoder.Config):
    query: PropertyEncoder.Query = PropertyEncoder.Query(
        label="EulerPrecise",
    )
    in_dim: int = 3
