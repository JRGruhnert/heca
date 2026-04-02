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
from src.states.addons.prepro_flip import (
    FlipStatePreprocessor,
    FlipStatePreprocessorConfig,
)
from src.states.logic.condition import Condition, ConditionConfig
from src.states.addons.state_preprocessor import (
    StatePreprocessor,
    StatePreprocessorConfig,
)
from src.states.rulers.ruler import Ruler, RulerConfig
from src.states.state import StateConfig, State
from src.states.value_handler.value_handler import ValueHandler, ValueHandlerConfig


def select_states(configs: Sequence[StateConfig]) -> list[State]:
    """Create states from configs - simple factory function"""
    return [State(config) for config in configs]


def select_skills(configs: Sequence[TreeNodeConfig]) -> list[TreeNode]:
    """Create skills from configs - simple factory function"""
    return [TreeNode(config) for config in configs]


def select_value_handler(config: ValueHandlerConfig) -> ValueHandler:
    """Create normalizer from config - simple factory function"""
    if isinstance(config, LinearValueConfig):
        return LinearValue(config)
    elif isinstance(config, IdentityValueConfig):
        return IdentityValue(config)
    elif isinstance(config, QuaternionValueConfig):
        return QuaternionValue(config)
    else:
        raise NotImplementedError(f"Unknown config.")


def select_distance(config: RulerConfig) -> Ruler:
    """Create distance condition from config - simple factory function"""
    if isinstance(config, BinaryRulerConfig):
        return ScalarDistance(config)
    elif isinstance(config, EuclideanRulerConfig):
        return EuclideanRuler(config)
    elif isinstance(config, FlipRulerConfig):
        return FlipRuler(config)
    elif isinstance(config, AngularRulerConfig):
        return AngularDistance(config)
    else:
        raise ValueError(f"Unknown config.")


def select_state_preprocessor(config: StatePreprocessorConfig) -> StatePreprocessor:
    """Create state preprocessor from config - simple factory function"""
    # if isinstance(config, SomeStatePreprocessorConfig):
    #     return SomeStatePreprocessor(config)
    raise NotImplementedError(f"Unknown config.")


def select_conditions(cons: dict[str, ConditionConfig]) -> dict[str, Condition]:
    """Create condition from config - simple factory function"""
    conditions = {}
    for key, config in cons.items():
        if isinstance(config, ValueDistanceConfig):
            conditions[key] = select_distance(config)
        elif isinstance(config, ValueHandlerConfig):
            conditions[key] = select_value_handler(config)
        else:
            raise NotImplementedError(f"Unknown config.")
    return conditions


def select_operator(config: NodeOperatorConfig) -> NodeOperator:
    """Create operator from config - simple factory function"""
    if isinstance(config, TapasOperatorConfig):
        return TapasLeafOperator(config)
    else:
        raise NotImplementedError(f"Unknown config.")


def select_parameter(config: NodeParameterConfig) -> NodeParameter:
    """Create operator loader from config - simple factory function"""
    if isinstance(config, TapasParameterConfig):
        return TapasParameter(config)
    else:
        raise NotImplementedError(f"Unknown config.")


def select_networker(config: NodeNetworkerConfig) -> NodeNetworker:
    """Create networker from config - simple factory function"""
    if isinstance(config, TapasNetworkerConfig):
        return TapasNetworker(config)
    else:
        raise NotImplementedError(f"Unknown config.")


def select_eval_condition(config: ValueEvaluationConfig) -> Evaluation:
    """Create eval condition from config - simple factory function"""
    if isinstance(config, ThresholdEvaluationConfig):
        return ThresholdEvaluation(config)
    else:
        raise NotImplementedError(f"Unknown config.")


def select_addon(config: StatePreprocessorConfig) -> StatePreprocessor:
    """Create addon from config - simple factory function"""
    if isinstance(config, FlipStatePreprocessorConfig):
        return FlipStatePreprocessor(config)
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
    if isinstance(config, PPOAgentConfig):
        return PPOAgent(config, buffer_module, storage_module)
    elif isinstance(config, SearchTreeAgentConfig):
        return SearchTreeAgent(config, storage_module, buffer_module)
    elif isinstance(config, HumanAgentConfig):
        return HumanAgent(config, storage_module, buffer_module)
    else:
        raise ValueError(f"Unknown agent type: {type(config)}")


def select_evaluator(
    config: EvaluatorConfig,
    storage: Storage,
) -> Evaluator:
    """Create reward module from config - simple factory function"""
    if isinstance(config, DenseEvaluatorConfig):
        return DenseEvaluator(config, storage)
    elif isinstance(config, Dense2EvaluatorConfig):
        return Dense2Evaluator(config, storage)
    elif isinstance(config, Dense3EvaluatorConfig):
        return Dense3Evaluator(config, storage)
    elif isinstance(config, SparseEvaluatorConfig):
        return SparseEvaluator(config, storage)
    else:
        raise ValueError(f"Unknown evaluator type: {type(config)}")


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
