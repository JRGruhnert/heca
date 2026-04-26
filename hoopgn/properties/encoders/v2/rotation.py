from dataclasses import dataclass

from hoopgn.properties.encoders.encoder import PropertyEncoder


class RotationEncoder(PropertyEncoder):
    @dataclass(kw_only=True)
    class Query(PropertyEncoder.Query):
        label: str = "rotation"

    @dataclass(kw_only=True)
    class Config(PropertyEncoder.Config):
        in_dim: int = 4
