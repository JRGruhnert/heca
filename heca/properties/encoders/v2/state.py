from dataclasses import dataclass

from heca.properties.encoders.encoder import PropertyEncoder


class StateEncoder(PropertyEncoder):
    @dataclass(kw_only=True)
    class Config(PropertyEncoder.Config):
        label: str = "state"
        in_dim: int = 1
