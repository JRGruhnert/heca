from dataclasses import dataclass

from heca.agents.agent import Agent
from heca.misc.base import Configurable
from heca.entities.precon import Precon
from heca.misc.td import TDScene


class Clusterer(Configurable):
    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        pass

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def __call__(self, x: TDScene, y: TDScene) -> TDScene:
        raise NotImplementedError()

    def compress(
        self, precons: list[Precon], postcons: list[Postcon]
    ) -> list[list[Entity]]:
        raise NotImplementedError()

    def merge(self, clusters: list[list[Entity]]) -> list[Entity]:
        raise NotImplementedError()


# Skill 1
# Precon
# 1 to 1
# Postcon

# Skill 2
# Precon
# 1 to 1
# Postcon

# Skill 3
# Precon
# 1 to 1
# Postcon

# Skill 4
# Precon
# 1 to 1
# Postcon

# Heca 1 (no compress)
# Precon
# 4 to 4
# Postcon

# Heca 2 (compress)
# Precon
# 2 to 2
# Postcon

# TDEntity


# Pre 1 -> Post 1
# current + pre -> (goal parent, current, pre) + post
# -> Construct sub goal
# sub goal -> cuurent(where no entity) + goal(where entity) + pre(sample)
# sub goal created equally like graph below:
# Every entity that is not in post -> current as replacement
# Every entity that is in post -> pre off other + goal where not pre of other
# Ever entity that is in post -> goal


# New Precon
# pos agg
# rot agg
# button0 1, 2 -> 1, 3
# button1 1, 2 -> 1, 2

# Skill 1
# pos a1        # pos a2
# rot a1        # rot a2
# button0 1     # button0 2

# Skill 2
# pos b1         # pos b2
# rot b1         # rot b2
# button0 2      # button0 1

# skill 3
# pos c1
# rot c1
# button1 on

# Skill 4
# pos d1
# pos d1
# button1 off

# Rule1 a-b & b-a => ab-ab
# Rule2 a-b & a-c => a-bc
# Rule3 a-b & c-b => ac->b

# Rule x a-b & c-d => ac-bd
