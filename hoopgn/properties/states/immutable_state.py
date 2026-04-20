from dataclasses import dataclass, field
import torch

from hoopgn.properties.states.state import State


class ImmutableState(State):
    @dataclass(kw_only=True)
    class Config(State.Config):
        values: set[str] = field(default_factory=set)

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.make_zeros()

    def label(self, x: torch.Tensor) -> str:
        raise NotImplementedError()
