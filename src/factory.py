from collections.abc import Sequence

from src.agents.agent import Agent, AgentConfig
from src.agents.human import HumanAgent, HumanAgentConfig
from src.agents.ppo import PPOAgent, PPOAgentConfig
from src.agents.search import SearchTreeAgent, SearchTreeAgentConfig
from src.environments.environment import Environment, EnvironmentConfig
from src.modules.buffer import Buffer
from src.modules.evaluators.dense import DenseEvaluator, DenseEvaluatorConfig
from src.modules.evaluators.dense2 import Dense2Evaluator, Dense2EvaluatorConfig
from src.modules.evaluators.dense3 import Dense3Evaluator, Dense3EvaluatorConfig
from src.modules.evaluators.evaluator import Evaluator, EvaluatorConfig
from src.modules.evaluators.sparse import SparseEvaluator, SparseEvaluatorConfig
from src.modules.storage import Storage
from src.experiments.experiment import Experiment, ExperimentConfig
from src.experiments.pepr import PePrExperiment, PePrConfig
from src.experiments.skill_check import SkillCheckExperiment, SkillCheckExperimentConfig
from src.environments.calvin import (
    CalvinEnvironment,
    CalvinEnvironmentConfig,
    CalvinEnvironmentConfig,
)

from src.objects.properties.property import State, StateConfig
from src.objects.properties.value_handler.evaluators.area_evaluator import (
    AreaEvaluator,
    AreaEvaluatorConfig,
)
from src.objects.properties.value_handler.evaluators.evaluator import (
    StateEvaluator,
    StateEvaluatorConfig,
)
from src.objects.properties.value_handler.evaluators.ignore_evaluator import (
    IgnoreEvaluator,
    IgnoreEvaluatorConfig,
)
from src.objects.properties.value_handler.evaluators.threshold_evaluator import (
    ThresholdEvaluator,
    ThresholdEvaluatorConfig,
)
from src.objects.properties.value_handler.parameters.parameter import (
    StateParameter,
    StateParameterConfig,
)
from src.objects.properties.value_handler.rulers.binary_ruler import (
    BinaryRulerConfig,
    BinaryRuler,
)
from src.objects.properties.value_handler.rulers.euclidean_ruler import (
    EuclideanRuler,
    EuclideanRulerConfig,
)
from src.objects.properties.value_handler.rulers.flip_ruler import (
    FlipRuler,
    FlipRulerConfig,
)
from src.objects.properties.value_handler.rulers.angular_ruler import (
    AngularRuler,
    AngularRulerConfig,
)
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
from src.skills.tree.networker import NodeNetworker, NodeNetworkerConfig
from src.skills.tree.node import TreeNode, TreeNodeConfig
from src.skills.tree.operator import NodeOperator, NodeOperatorConfig
from src.skills.tree.parameter import NodeParameter, NodeParameterConfig


STATE_RULER_BUILDERS = {
    BinaryRulerConfig: lambda config: BinaryRuler(config),
    EuclideanRulerConfig: lambda config: EuclideanRuler(config),
    FlipRulerConfig: lambda config: FlipRuler(config),
    AngularRulerConfig: lambda config: AngularRuler(config),
}

AGENT_BUILDERS = {
    PPOAgentConfig: lambda config, storage, buffer: PPOAgent(config, buffer, storage),
    SearchTreeAgentConfig: lambda config, storage, buffer: SearchTreeAgent(
        config,
        storage,
        buffer,
    ),
    HumanAgentConfig: lambda config, storage, buffer: HumanAgent(
        config,
        storage,
        buffer,
    ),
}

EVALUATOR_BUILDERS = {
    DenseEvaluatorConfig: lambda config, storage: DenseEvaluator(config, storage),
    Dense2EvaluatorConfig: lambda config, storage: Dense2Evaluator(config, storage),
    Dense3EvaluatorConfig: lambda config, storage: Dense3Evaluator(config, storage),
    SparseEvaluatorConfig: lambda config, storage: SparseEvaluator(config, storage),
}


def select_states(configs: Sequence[StateConfig]) -> list[State]:
    """Create states from configs - simple factory function"""
    return [State(config) for config in configs]


def select_skills(configs: Sequence[TreeNodeConfig]) -> list[TreeNode]:
    """Create skills from configs - simple factory function"""
    return [TreeNode(config) for config in configs]


def select_eval_condition(config: StateEvaluatorConfig) -> StateEvaluator:
    """Create eval condition from config - simple factory function"""
    if isinstance(config, ThresholdEvaluatorConfig):
        return ThresholdEvaluator(config)
    elif isinstance(config, IgnoreEvaluatorConfig):
        return IgnoreEvaluator(config)
    elif isinstance(config, AreaEvaluatorConfig):
        return AreaEvaluator(config)
    else:
        raise NotImplementedError(f"Unknown config.")


def select_parameter(config: StateParameterConfig) -> StateParameter:
    """Create addon from config - simple factory function"""
    if isinstance(config, FlipStateParameterConfig):
        return FlipStateParameter(config)
    else:
        raise NotImplementedError(f"Unknown config.")


def select_value_handler(config: ValueHandlerConfig) -> ValueHandler:
    if isinstance(config, LinearValueConfig):
        return LinearValue(config)
    elif isinstance(config, IdentityValueConfig):
        return IdentityValue(config)
    elif isinstance(config, QuaternionValueConfig):
        return QuaternionValue(config)
    else:
        raise NotImplementedError(f"Unknown config.")


def select_state_ruler(config: RulerConfig) -> Ruler:
    """Create distance condition from config - simple factory function"""
    builder = STATE_RULER_BUILDERS.get(type(config))
    if builder is None:
        raise ValueError(f"Unknown config type: {type(config)}")
    return builder(config)


def select_state_validator(config: ValidatorConfig) -> Validator:
    if isinstance(config, AreaValidatorConfig):
        return AreaValidator(config)
    else:
        raise NotImplementedError(f"Unknown config.")


def select_state_parameter(config: StateParameterConfig) -> StateParameter:

    raise NotImplementedError(f"Unknown config.")


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


def select_experiment(
    config: ExperimentConfig,
    env: Environment,
    storage: Storage,
) -> Experiment:
    """Create experiment from config - simple factory function"""

    if isinstance(config, PePrConfig):
        return PePrExperiment(config, env, storage)
    elif isinstance(config, SkillCheckExperimentConfig):
        return SkillCheckExperiment(config, env, storage)
    else:
        raise ValueError(f"Unknown experiment type: {type(config)}")


def select_agent(
    config: AgentConfig,
    storage_module: Storage,
    buffer_module: Buffer,
) -> Agent:
    """Create agent from config - simple factory function"""
    builder = AGENT_BUILDERS.get(type(config))
    if builder is None:
        raise ValueError(f"Unknown agent type: {type(config)}")
    return builder(config, storage_module, buffer_module)


def select_evaluator(
    config: EvaluatorConfig,
    storage: Storage,
) -> Evaluator:
    """Create reward module from config - simple factory function"""
    builder = EVALUATOR_BUILDERS.get(type(config))
    if builder is None:
        raise ValueError(f"Unknown evaluator type: {type(config)}")
    return builder(config, storage)


def select_environment(
    config: EnvironmentConfig,
    evaluator: Evaluator,
    storage: Storage,
) -> Environment:
    """Create environment from config - simple factory function"""
    if isinstance(config, CalvinEnvironmentConfig):
        return CalvinEnvironment(
            config,
            evaluator,
            storage,
        )
    else:
        raise ValueError(f"Unknown environment type: {type(config)}")
