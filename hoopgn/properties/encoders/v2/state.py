from dataclasses import dataclass

from hoopgn.properties.encoders.encoder import PropertyEncoder


class StateEncoder(PropertyEncoder):
    @dataclass(kw_only=True)
    class Query(PropertyEncoder.Query):
        label: str = "state"

    @dataclass(kw_only=True)
    class Config(PropertyEncoder.Config):
        in_dim: int = 1
