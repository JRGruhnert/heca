from dataclasses import dataclass

from src.observation.observation import StateValueDict

from src.skills.networker import NodeNetworker, NodeNetworkerConfig


@dataclass
class TreeNetworkerConfig(NodeNetworkerConfig):
    pass


class TreeNetworker(NodeNetworker):
    def __init__(self, config: TreeNetworkerConfig):
        self.config = config

    def reset(self, goal: StateValueDict):
        """Prepare the networker for execution. Before each use."""
        raise NotImplementedError("Subclasses must implement method.")
