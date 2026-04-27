from dataclasses import dataclass

from heca.properties.encoders.encoder import PropertyEncoder


class BoolEncoder(PropertyEncoder):
    @dataclass(kw_only=True)
    class Query(PropertyEncoder.Query):
        label: str = "Bool"

    @dataclass(kw_only=True)
    class Config(PropertyEncoder.Config):
        in_dim: int = 1
        hidden_dim: int = 8
