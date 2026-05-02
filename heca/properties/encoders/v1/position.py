from dataclasses import dataclass

from heca.properties.encoders.encoder import PropertyEncoder


class PositionEncoder(PropertyEncoder):

    @dataclass(frozen=True, kw_only=True)
    class Query(PropertyEncoder.Query):
        label: str = "EulerPrecise"

    @dataclass(kw_only=True)
    class Config(PropertyEncoder.Config):
        in_dim: int = 3
