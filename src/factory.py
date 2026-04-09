from collections.abc import Sequence

from src.objects.properties.property import State, StateConfig

from src.skills.tree.leafs.tapas.tapas_networker import (
    TapasNetworker,
    TapasNetworkerConfig,
)
from src.skills.tree.leafs.tapas.tapas_operator import (
    TapasLeafOperator,
    TapasOperatorConfig,
)
from src.skills.tree.leafs.tapas.tapas_parameter import (
    TapasParameter,
    TapasParameterConfig,
)
from src.skills.networker import NodeNetworker, NodeNetworkerConfig
from src.skills.node import TreeNode, TreeNodeConfig
from src.skills.operator import NodeOperator, NodeOperatorConfig
from src.skills.parameter import NodeParameter, NodeParameterConfig


def select_states(configs: Sequence[StateConfig]) -> list[State]:
    """Create states from configs - simple factory function"""
    return [State(config) for config in configs]


def select_skills(configs: Sequence[TreeNodeConfig]) -> list[TreeNode]:
    """Create skills from configs - simple factory function"""
    return [TreeNode(config) for config in configs]


def select_node_parameter(config: NodeParameterConfig) -> NodeParameter:
    """Create operator loader from config - simple factory function"""
    if isinstance(config, TapasParameterConfig):
        return TapasParameter(config)
    else:
        raise NotImplementedError(f"Unknown config.")


def select_operator(config: NodeOperatorConfig) -> NodeOperator:
    """Create operator from config - simple factory function"""
    if isinstance(config, TapasOperatorConfig):
        return TapasLeafOperator(config)
    else:
        raise NotImplementedError(f"Unknown config.")


def select_networker(config: NodeNetworkerConfig) -> NodeNetworker:
    """Create networker from config - simple factory function"""
    if isinstance(config, TapasNetworkerConfig):
        return TapasNetworker(config)
    else:
        raise NotImplementedError(f"Unknown config.")
