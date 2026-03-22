from cli.commands.explain import ExplainManagerConfig
from src.agents.ppo import PPOAgentConfig
from src.environments.calvin import CalvinEnvironmentConfig
from src.environments.environment import EnvironmentConfig
from src.modules.buffer import BufferConfig
from src.modules.evaluators.dense3 import Dense3EvaluatorConfig
from src.modules.logger import LogMode, LoggerConfig
from src.modules.storage import StorageConfig
from src.experiments.pepr import PePrConfig
from src.networks.baseline import BaselineNetworkConfig
from src.networks.gnn import GraphNetworkConfig
from src.networks.network import NetworkConfig

evaluator = Dense3EvaluatorConfig(
    success_reward=25.0,
    max_progress_reward=1.0,
    step_penalty=-0.002,
    add_monotonic_reward=True,
)


def experiment_config(p_empty: float, p_rand: float) -> PePrConfig:
    return PePrConfig(
        p_empty=p_empty,
        p_rand=p_rand,
    )


def storage_config(
    network: str,
    prefix_tag: str,
    state_set_tag: str,
    skill_set_tag: str,
) -> StorageConfig:
    return StorageConfig(
        used_skills=skill_set_tag,
        used_states=state_set_tag,
        eval_states=skill_set_tag,
        tag=f"{prefix_tag}_{state_set_tag}_{skill_set_tag}",
        network=network,
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
    state_set_tag: str,
    skill_set_tag: str,
) -> LoggerConfig:
    return LoggerConfig(
        mode=mode,
        wandb_tag=f"{network_tag}_{prefix_tag}_{state_set_tag}_{skill_set_tag}",
    )


def network_config(
    checkpoint_name: str | None, is_baseline: bool, explain_mode: bool
) -> NetworkConfig:
    checkpoint_path = (
        f"results/{is_baseline}/{checkpoint_name}/model_cp_best.pth"
        if checkpoint_name
        else None
    )
    return (
        GraphNetworkConfig(
            checkpoint_path=checkpoint_path,
            explain_mode=explain_mode,
        )
        if not is_baseline
        else BaselineNetworkConfig(
            checkpoint_path=checkpoint_path,
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
