from dataclasses import dataclass

from heca.properties.encoders.encoder import PropertyEncoder


class RotationEncoder(PropertyEncoder):
    @dataclass(frozen=True, kw_only=True)
    class Query(PropertyEncoder.Query):
        label: str = "rotation"

    @dataclass(kw_only=True)
    class Config(PropertyEncoder.Config):
        in_dim: int = 4
