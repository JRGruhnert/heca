from dataclasses import dataclass

from heca.properties.encoders.encoder import PropertyEncoder


class RangeEncoder(PropertyEncoder):

    @dataclass(frozen=True, kw_only=True)
    class Query(PropertyEncoder.Query):
        label: str = "Range"

    @dataclass(kw_only=True)
    class Config(PropertyEncoder.Config):
        in_dim: int = 1
        hidden_dim: int = 8
