from dataclasses import dataclass

from hoopgn.misc.state import State


class CalvinAreaState(State):
    @dataclass(kw_only=True)
    class Config(State.Config):
        values: set[str] = {
            "empty",
            "red",
            "green",
            "blue",
            "yellow",
        }

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
