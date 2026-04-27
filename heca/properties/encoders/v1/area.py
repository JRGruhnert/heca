from dataclasses import dataclass

from heca.properties.encoders.encoder import PropertyEncoder


class AreaEncoder(PropertyEncoder):
    @dataclass(kw_only=True)
    class Query(PropertyEncoder.Query):
        label: str = "AreaEuler"

    @dataclass(kw_only=True)
    class Config(PropertyEncoder.Config):
        in_dim: int = 6
