from dataclasses import dataclass

from src.skills.tree.networker import NodeNetworker, NodeNetworkerConfig


@dataclass
class TapasNetworkerConfig(NodeNetworkerConfig):
    pass


class TapasNetworker(NodeNetworker):
    def __init__(self, config: TapasNetworkerConfig):
        super().__init__(config)
        self.config = config

    def __call__(self, start):
        "TODO: implement"
        raise NotImplementedError()

    def reset(self, goal):
        "TODO: implement"
        raise NotImplementedError()
