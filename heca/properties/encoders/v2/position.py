from dataclasses import dataclass

from heca.properties.encoders.encoder import PropertyEncoder


class PositionEncoder(PropertyEncoder):
    @dataclass(kw_only=True)
    class Config(PropertyEncoder.Config):
        label: str = "position"
        in_dim: int = 3
