from dataclasses import dataclass
from tensordict import TensorDict

from hoopgn.base import ConfigurableClass


class Converter(ConfigurableClass):
    @dataclass(kw_only=True)
    class Config(ConfigurableClass.Config):
        label: str

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def __call__(self, observation) -> TensorDict:
        raise NotImplementedError(
            "ObservationConverter __call__ method not implemented yet."
        )
