from dataclasses import dataclass

from hoopgn.properties.encoders.encoder import PropertyEncoder


class DomainEncoder(PropertyEncoder):
    @dataclass(kw_only=True)
    class Query(PropertyEncoder.Query):
        label: str = "domain"

    @dataclass(kw_only=True)
    class Config(PropertyEncoder.Config):
        in_dim: int = 1
        hidden_dim: int = 8
