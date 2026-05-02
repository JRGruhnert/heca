from dataclasses import dataclass, field

from heca.misc.area import Area
from heca.misc.state import State


class CalvinAreaState(State):
    @dataclass(kw_only=True)
    class Config(State.Config):
        area: Area.Config
        values: set[str] = field(init=False)

        def __post_init__(self):
            assert len(self.values) > 0, "At least one value must be defined."
            self.values = self.area.labels

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
