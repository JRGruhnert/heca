from dataclasses import dataclass

from heca.properties.encoders.encoder import PropertyEncoder


class FlipEncoder(PropertyEncoder):
    @dataclass(kw_only=True)
    class Query(PropertyEncoder.Query):
        label: str = "Flip"

    @dataclass(kw_only=True)
    class Config(PropertyEncoder.Config):
        in_dim: int = 1
        hidden_dim: int = 8
