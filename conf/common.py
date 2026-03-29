from collections.abc import Sequence

from conf.state_sets import STATES_SETS, StateSet
from conf.leaf_sets import SKILL_SETS, LeafSet
from src.agents.ppo import PPOAgentConfig
from src.environments.calvin import CalvinEnvironmentConfig
from src.environments.environment import EnvironmentConfig
from src.modules.evaluators.dense3 import Dense3EvaluatorConfig
from src.modules.logger import LogMode, LoggerConfig
from src.modules.storage import StorageConfig
from src.experiments.pepr import PePrConfig
from src.networks.baseline import BaselineNetworkConfig
from src.networks.gnn import GraphNetworkConfig
from src.networks.network import NetworkConfig
from src.skills.tree.leafs.leaf import LeafConfig
from src.states.state import StateConfig

evaluator = Dense3EvaluatorConfig(
    success_reward=25.0,
    max_progress_reward=1.0,
    step_penalty=-0.002,
    add_monotonic_reward=True,
)


def experiment_config(p_empty: float, p_rand: float, min_steps) -> PePrConfig:
    return PePrConfig(
        p_empty=p_empty,
        p_rand=p_rand,
        min_steps=min_steps,
    )


def storage_config(
    prefix_tag: str,
    state_set: StateSet,
    skill_set: LeafSet,
) -> StorageConfig:
    return StorageConfig(
        skills=skill_configs(skill_set),
        states_network=state_configs(state_set),
        states_eval=state_configs(state_set),
        tag=f"{prefix_tag}_{state_set.value}_{skill_set.value}",
    )


def agent_config(network: NetworkConfig, eval: bool) -> PPOAgentConfig:
    return PPOAgentConfig(
        network=network,
        eval=eval,
        max_batches=750,
        early_stop_patience=50,
        min_batches=250,
        use_ema_for_early_stopping=False,
    )


def logger_config(
    mode: LogMode,
    network_tag: str,
    prefix_tag: str,
    state_set: StateSet,
    skill_set: LeafSet,
) -> LoggerConfig:
    return LoggerConfig(
        mode=mode,
        wandb_tag=f"{network_tag}_{prefix_tag}_{state_set.value}_{skill_set.value}",
    )


def network_config(
    is_gnn: bool,
    checkpoint_name: str | None,
    explain_mode: bool,
    skill_count: int,
    state_count: int,
) -> NetworkConfig:
    network_name = "gnn" if is_gnn else "baseline"
    checkpoint_path = (
        f"results/{network_name}/{checkpoint_name}/model_cp_best.pth"
        if checkpoint_name
        else None
    )
    return (
        GraphNetworkConfig(
            checkpoint_path=checkpoint_path,
            explain_mode=explain_mode,
            skill_count=skill_count,
            state_count=state_count,
        )
        if is_gnn
        else BaselineNetworkConfig(
            checkpoint_path=checkpoint_path,
            skill_count=skill_count,
            state_count=state_count,
        )
    )


def environment_config(
    environment_tag="calvin", render: bool = False
) -> EnvironmentConfig:
    if environment_tag == "calvin":
        return CalvinEnvironmentConfig(render=render)
    else:
        raise ValueError(f"Unknown environment tag: {environment_tag}")


def evaluator_config(evaluator_tag: str = "dense3") -> Dense3EvaluatorConfig:
    if evaluator_tag == "dense3":
        return Dense3EvaluatorConfig(
            success_reward=25.0,
            max_progress_reward=1.0,
            step_penalty=-0.002,
            add_monotonic_reward=True,
        )
    else:
        raise ValueError(f"Unknown evaluator tag: {evaluator_tag}")


def skill_configs(skill_set: LeafSet) -> Sequence[LeafConfig]:
    return SKILL_SETS[skill_set]


def state_configs(state_set: StateSet) -> Sequence[StateConfig]:
    return STATES_SETS[state_set]
