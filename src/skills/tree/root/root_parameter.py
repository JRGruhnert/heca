from dataclasses import dataclass

from src.skills.parameter import NodeParameter, NodeParameterConfig


@dataclass
class TreeParameterConfig(NodeParameterConfig):
    pass


class TreeParameter(NodeParameter):
    def __init__(self, config: TreeParameterConfig):
        self.config = config
