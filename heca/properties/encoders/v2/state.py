from dataclasses import dataclass

from heca.properties.encoders.encoder import PropertyEncoder


class StateEncoder(PropertyEncoder):
    @dataclass(frozen=True, kw_only=True)
    class Query(PropertyEncoder.Query):
        label: str = "state"

    @dataclass(kw_only=True)
    class Config(PropertyEncoder.Config):
        in_dim: int = 1

    @dataclass(frozen=True, kw_only=True)
    class File(PropertyEncoder.Location):
        folder: str = "state"
        ending: str = ".pt"
