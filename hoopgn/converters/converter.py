from dataclasses import dataclass
from tensordict import TensorDict
from hoopgn.misc.td import TDEntities
from hoopgn.misc.classes import ConfigurableClass


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


class HoopConverter(Converter):
    def __call__(self, obs) -> TDEntities:
        raise NotImplementedError()


class LeafConverter(Converter):
    def __call__(self, obs) -> TensorDict:
        raise NotImplementedError()
