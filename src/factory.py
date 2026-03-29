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

from src.skills.tree.leafs.leaf import Leaf, LeafConfig
from src.states.logic.addons.addon_flip import (
    FlipStatePreprocessor,
    FlipStatePreprocessorConfig,
)
from src.states.logic.condition import Condition, ConditionConfig
from src.states.logic.rotation.quaternion_value_cnd import (
    QuaternionValueCondition,
    QuaternionValueConditionConfig,
)
from src.states.logic.distance import Distance, DistanceConfig
from src.states.logic.eval_cnd import EvalCondition, EvalConditionConfig
from src.states.logic.scalars.switch_distance_cnd import (
    SwitchDistanceCondition,
    SwitchDistanceConditionConfig,
)
from src.states.logic.identity.identity_value_cnd import (
    IdentityValue,
    IdentityValueConfig,
)
from src.states.logic.linear.linear_value_cnd import (
    LinearValueNormalizer,
    LinearValueNormalizerConfig,
)
from src.states.logic.location.euclidean_distance_cnd import (
    EuclideanDistanceCondition,
    EuclideanDistanceConditionConfig,
)
from src.states.logic.state_preprocessor import (
    StatePreprocessor,
    StatePreprocessorConfig,
)
from src.states.logic.thresholds.threshold_eval_cnd import (
    ThresholdEvalCondition,
    ThresholdEvalConditionConfig,
)
from src.states.logic.scalars.range_distance_cnd import (
    RangeDistanceCondition,
    RangeDistanceConditionConfig,
)
from src.states.logic.rotation.quaternion_distance_cnd import (
    QuaternionDistanceCondition,
    QuaternionDistanceConditionConfig,
)
from src.states.logic.value_cnd import ValueCondition, ValueConditionConfig
from src.states.state import StateConfig, State


def select_states(configs: Sequence[StateConfig]) -> list[State]:
    """Create states from configs - simple factory function"""
    return [State(config) for config in configs]


def select_skills(configs: Sequence[LeafConfig]) -> list[Leaf]:
    """Create skills from configs - simple factory function"""
    return [Leaf(config) for config in configs]


def select_value_condition(config: ValueConditionConfig) -> ValueCondition:
    """Create normalizer from config - simple factory function"""
    if isinstance(config, LinearValueNormalizerConfig):
        return LinearValueNormalizer(config)
    elif isinstance(config, IdentityValueConfig):
        return IdentityValue(config)
    elif isinstance(config, QuaternionValueConditionConfig):
        return QuaternionValueCondition(config)
    else:
        raise NotImplementedError(f"Unknown config.")


def select_distance_condition(config: DistanceConfig) -> Distance:
    """Create distance condition from config - simple factory function"""
    if isinstance(config, RangeDistanceConditionConfig):
        return RangeDistanceCondition(config)
    elif isinstance(config, EuclideanDistanceConditionConfig):
        return EuclideanDistanceCondition(config)
    elif isinstance(config, SwitchDistanceConditionConfig):
        return SwitchDistanceCondition(config)
    elif isinstance(config, QuaternionDistanceConditionConfig):
        return QuaternionDistanceCondition(config)
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
        if isinstance(config, DistanceConfig):
            conditions[key] = select_distance_condition(config)
        elif isinstance(config, ValueConditionConfig):
            conditions[key] = select_value_condition(config)
        else:
            raise NotImplementedError(f"Unknown config.")
    return conditions


def select_eval_condition(config: EvalConditionConfig) -> EvalCondition:
    """Create eval condition from config - simple factory function"""
    if isinstance(config, ThresholdEvalConditionConfig):
        return ThresholdEvalCondition(config)
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
