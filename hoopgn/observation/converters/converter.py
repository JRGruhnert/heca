from dataclasses import dataclass
from tensordict import TensorDictBase


@dataclass(kw_only=True)
class ConverterConfig:
    label: str


class Converter:
    def __init__(self, config: ConverterConfig):
        self.config = config

    def __call__(self, observation) -> TensorDictBase:
        raise NotImplementedError(
            "ObservationConverter __call__ method not implemented yet."
        )
