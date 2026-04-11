from conf.property_sets import OBJECT_SETS
from conf.skill_sets import SKILL_SETS
from hoopgn.agents.ppo import PPOAgentConfig
from hoopgn.environments.calvin import CalvinEnvironmentConfig
from hoopgn.environments.environment import EnvironmentConfig
from hoopgn.evaluators.dense3 import Dense3EvaluatorConfig
from hoopgn.evaluators.evaluator import EvaluatorConfig
from hoopgn.experiments.experiment import ExperimentConfig
from hoopgn.logger import LogMode, LoggerConfig
from hoopgn.storage import StorageConfig
from hoopgn.experiments.noise_experiment import NoiseExperimentConfig
from hoopgn.networks.baseline import BaselineNetworkConfig
from hoopgn.networks.gnn import GraphNetworkConfig
from hoopgn.networks.network import NetworkConfig


def evaluator_config(
    tag: str,
    state_eval_tag: str,
    state_network_tag: str,
) -> EvaluatorConfig:
    if tag == "dense3":
        return Dense3EvaluatorConfig(
            success_reward=25.0,
            max_progress_reward=1.0,
            step_penalty=-0.002,
            add_monotonic_reward=True,
            states_network=OBJECT_SETS[state_network_tag],
            states_eval=OBJECT_SETS[state_eval_tag],
        )
    else:
        raise ValueError(f"Unknown evaluator tag: {tag}")


def experiment_config(
    p_empty: float,
    p_rand: float,
    min_steps: int,
    skill_set_tag: str,
    state_eval_tag: str,
    state_network_tag: str,
    environment_tag: str = "calvin",
    evaluator_tag: str = "dense3",
) -> ExperimentConfig:
    return NoiseExperimentConfig(
        p_empty=p_empty,
        p_rand=p_rand,
        min_steps=min_steps,
        environment=environment_config(environment_tag),
        evaluator=evaluator_config(evaluator_tag, state_eval_tag, state_network_tag),
        skills=SKILL_SETS[skill_set_tag],
    )


def storage_config(
    prefix_tag: str,
    state_set_tag: str,
    skill_set_tag: str,
) -> StorageConfig:
    return StorageConfig(
        skills=SKILL_SETS[skill_set_tag],
        states_network=OBJECT_SETS[state_set_tag],
        states_eval=OBJECT_SETS[state_set_tag],
        tag=f"{prefix_tag}_{state_set_tag}_{skill_set_tag}",
    )


def ppo_default_config(
    network: NetworkConfig,
    eval: bool,
    batch_size: int,
) -> PPOAgentConfig:
    return PPOAgentConfig(
        network=network,
        batch_size=batch_size,
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
            dim_skill=skill_count,
            dim_state=state_count,
        )
        if is_gnn
        else BaselineNetworkConfig(
            checkpoint_path=checkpoint_path,
            dim_skill=skill_count,
            dim_state=state_count,
        )
    )


def environment_config(tag="calvin") -> EnvironmentConfig:
    if tag == "calvin":
        return CalvinEnvironmentConfig()
    else:
        raise ValueError(f"Unknown environment tag: {tag}")
