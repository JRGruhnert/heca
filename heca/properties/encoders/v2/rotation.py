from dataclasses import dataclass

from heca.properties.encoders.encoder import PropertyEncoder


class RotationEncoder(PropertyEncoder):
    @dataclass(kw_only=True)
    class Config(PropertyEncoder.Config):
        label: str = "rotation"
        in_dim: int = 4
