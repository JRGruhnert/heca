from dataclasses import dataclass

from heca.properties.encoders.encoder import PropertyEncoder


class RotationEncoder(PropertyEncoder):
    @dataclass(frozen=True, kw_only=True)
    class Query(PropertyEncoder.Query):
        label: str = "rotation"

    @dataclass(kw_only=True)
    class Config(PropertyEncoder.Config):
        in_dim: int = 4

    @dataclass(frozen=True, kw_only=True)
    class File(PropertyEncoder.Location):
        folder: str = "rotation"
        ending: str = ".pt"
