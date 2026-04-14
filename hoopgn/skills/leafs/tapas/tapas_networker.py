from dataclasses import dataclass
from torch_geometric.data import Batch
from hoopgn.skills.skill_networker import SkillNetworker, SkillNetworkerConfig


@dataclass(kw_only=True)
class TapasNetworkerConfig(SkillNetworkerConfig):
    pass


class TapasNetworker(SkillNetworker):
    def __init__(self, config: TapasNetworkerConfig):
        super().__init__(config)
        self.config = config

    def __call__(self, x) -> Batch:
        "TODO: implement"
        return Batch()  # placeholder

    def reset(self, goal):
        "TODO: implement"
        pass
