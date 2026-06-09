from dataclasses import dataclass

from heca.properties.encoders.encoder import PropertyEncoder


class DomainEncoder(PropertyEncoder):
    @dataclass(kw_only=True)
    class Config(PropertyEncoder.Config):
        folder: str = "domain"
        in_dim: int = 1
        hidden_dim: int = 8
