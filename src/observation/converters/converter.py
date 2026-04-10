from dataclasses import dataclass


@dataclass(kw_only=True)
class ConverterConfig:
    pass


class Converter:
    def __init__(self, config: ConverterConfig):
        self.config = config

    def __call__(self, observation) -> dict[str, float]:
        raise NotImplementedError(
            "ObservationConverter __call__ method not implemented yet."
        )
