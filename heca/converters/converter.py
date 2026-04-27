from dataclasses import dataclass
from tensordict import TensorDict
from heca.misc.td import TDEntities
from heca.classes.config import Configurable


class Converter(Configurable):
    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        label: str

    def __call__(self, observation) -> TensorDict:
        raise NotImplementedError(
            "ObservationConverter __call__ method not implemented yet."
        )


class HecaConverter(Converter):
    @dataclass(kw_only=True)
    class Config(Converter.Config):
        pass

    def __call__(self, obs) -> TDEntities:
        raise NotImplementedError()


class LeafConverter(Converter):
    @dataclass(kw_only=True)
    class Config(Converter.Config):
        pass

    def __call__(self, obs) -> TensorDict:
        raise NotImplementedError()
