from abc import abstractmethod
from dataclasses import dataclass
from functools import cached_property
import numpy as np
from torch_geometric.data import Batch
from external.calvin_env_modified.build.lib.calvin_env_modified.envs.observation import (
    CalvinEnvObservation,
)
from src.factory import select_parameter, select_networker, select_operator
from src.observation.observation import StateValueDict
from src.skills.tree.networker import NodeNetworkerConfig
from src.skills.tree.operator import NodeOperatorConfig
from src.skills.tree.parameter import NodeParameterConfig
from src.states.logic.condition import Condition
from src.states.state import State


@dataclass
class TreeNodeConfig:
    label: str
    id: int
    childs: list[int]
    operator: NodeOperatorConfig
    parameter: NodeParameterConfig
    networker: NodeNetworkerConfig


class TreeNode:
    def __init__(self, config: TreeNodeConfig):
        self.config = config
        self.operator = select_operator(config.operator)
        self.network_handler = select_networker(config.networker)
        self.stamp = select_parameter(config.parameter)

    def reset(self, goal):
        self.operator.reset(goal)
        self.network_handler.reset(goal)
        # self.stamp.reset(goal)

    def build_network(self, x: StateValueDict, y: StateValueDict) -> Batch:
        return self.network_handler(x, y)

    def predict(self, *args, **kwargs) -> np.ndarray | None:
        return self.operator(*args, **kwargs)

    def predict_old(
        self, current: CalvinEnvObservation, states: list[State]
    ) -> np.ndarray | None:
        return self.operator(current, states)

    def load(self, x: StateValueDict, y: StateValueDict) -> float:
        return self.stamp(x, y)

    @cached_property
    def parameter(self) -> set[str]:
        return set(self.precons.keys()) | set(self.postcons.keys())

    @cached_property
    def precons(self) -> dict[str, Condition]:
        return self.operator.load_parameter_precons()

    @cached_property
    def postcons(self) -> dict[str, Condition]:
        return self.operator.load_parameter_postcons()
