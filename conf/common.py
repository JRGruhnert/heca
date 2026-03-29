from conf.calvin.states import MasterStateSet
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
from src.skills.skill import SkillConfig
from src.states.state import StateConfig
from src.variables import (
    SET_SLIDE,
    SET_BLUE,
    SET_RED,
    SET_PINK,
    SET_SR,
    SET_SRP,
    SET_SRPB,
    SET_SRPB,
)

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
    prefix_tag: str,
    state_set_tag: str,
    skill_set_tag: str,
) -> StorageConfig:
    return StorageConfig(
        used_skills=skill_set_tag,
        used_states=state_set_tag,
        eval_states=skill_set_tag,
        tag=f"{prefix_tag}_{state_set_tag}_{skill_set_tag}",
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


def skill_configs(tag: str) -> list[SkillConfig]:
    if tag == SET_SLIDE:
        return []
    else:
        raise ValueError("")


def state_configs(tag: str) -> list[StateConfig]:
    if tag == SET_SLIDE:
        return [
            MasterStateSet.ee_position,
            MasterStateSet.slide_position,
            MasterStateSet.drawer_position,
            MasterStateSet.button_position,
            MasterStateSet.led_position,
            MasterStateSet.ee_rotation,
            MasterStateSet.slide_rotation,
            MasterStateSet.drawer_rotation,
            MasterStateSet.button_rotation,
            MasterStateSet.led_rotation,
            MasterStateSet.ee_scalar,
            MasterStateSet.slide_scalar,
            MasterStateSet.drawer_scalar,
            MasterStateSet.button_scalar,
            MasterStateSet.led_rotation,
        ]
    elif tag == SET_BLUE:
        return []
    else:
        raise ValueError("")


def get_number_of_states_by_tag(tag: str) -> int:
    return len(STATES_BY_TAG.get(tag, []))


_S = {}
STATES_BY_TAG = {
    "Debug": [
        _S["ee_position"],
        _S["base__slide_position"],
        _S["base__drawer_position"],
        _S["base__button_position"],
        _S["led_position"],
        _S["block_red_position"],
        _S["block_blue_position"],
        _S["block_pink_position"],
        _S["ee_rotation"],
        _S["base__slide_rotation"],
        _S["base__drawer_rotation"],
        _S["base__button_rotation"],
        _S["led_rotation"],
        _S["block_red_rotation"],
        _S["block_blue_rotation"],
        _S["block_pink_rotation"],
        _S["ee_scalar"],
        _S["base__slide_scalar"],
        _S["base__drawer_scalar"],
        _S["base__button_scalar"],
        _S["block_red_scalar"],
        _S["block_blue_scalar"],
        _S["block_pink_scalar"],
    ],
    SET_SLIDE: [
        _S["ee_position"],
        _S["base__slide_position"],
        _S["base__drawer_position"],
        _S["base__button_position"],
        _S["led_position"],
        _S["ee_rotation"],
        _S["base__slide_rotation"],
        _S["base__drawer_rotation"],
        _S["base__button_rotation"],
        _S["led_rotation"],
        _S["ee_scalar"],
        _S["base__slide_scalar"],
        _S["base__drawer_scalar"],
        _S["base__button_scalar"],
    ],
    SET_RED: [
        _S["ee_position"],
        _S["ee_rotation"],
        _S["ee_scalar"],
        _S["base__drawer_position"],
        _S["base__drawer_rotation"],
        _S["base__drawer_scalar"],
        _S["base__button_position"],
        _S["base__button_rotation"],
        _S["base__button_scalar"],
        _S["led_position"],
        _S["led_rotation"],
        _S["block_red_position"],
        _S["block_red_rotation"],
        _S["block_red_scalar"],
    ],
    SET_BLUE: [
        _S["ee_position"],
        _S["ee_rotation"],
        _S["ee_scalar"],
        _S["base__drawer_position"],
        _S["base__drawer_rotation"],
        _S["base__drawer_scalar"],
        _S["base__button_position"],
        _S["base__button_rotation"],
        _S["base__button_scalar"],
        _S["led_position"],
        _S["led_rotation"],
        _S["block_blue_position"],
        _S["block_blue_rotation"],
        _S["block_blue_scalar"],
    ],
    SET_PINK: [
        _S["ee_position"],
        _S["ee_rotation"],
        _S["ee_scalar"],
        _S["base__drawer_position"],
        _S["base__drawer_rotation"],
        _S["base__drawer_scalar"],
        _S["base__button_position"],
        _S["base__button_rotation"],
        _S["base__button_scalar"],
        _S["led_position"],
        _S["led_rotation"],
        _S["block_pink_position"],
        _S["block_pink_rotation"],
        _S["block_pink_scalar"],
    ],
    SET_SR: [
        _S["ee_position"],
        _S["ee_rotation"],
        _S["ee_scalar"],
        _S["base__slide_position"],
        _S["base__slide_rotation"],
        _S["base__slide_scalar"],
        _S["base__drawer_position"],
        _S["base__drawer_rotation"],
        _S["base__drawer_scalar"],
        _S["base__button_position"],
        _S["base__button_rotation"],
        _S["base__button_scalar"],
        _S["led_position"],
        _S["led_rotation"],
        _S["block_red_position"],
        _S["block_red_rotation"],
        _S["block_red_scalar"],
    ],
    SET_SRP: [
        _S["ee_position"],
        _S["ee_rotation"],
        _S["ee_scalar"],
        _S["base__slide_position"],
        _S["base__slide_rotation"],
        _S["base__slide_scalar"],
        _S["base__drawer_position"],
        _S["base__drawer_rotation"],
        _S["base__drawer_scalar"],
        _S["base__button_position"],
        _S["base__button_rotation"],
        _S["base__button_scalar"],
        _S["led_position"],
        _S["led_rotation"],
        _S["block_red_position"],
        _S["block_red_rotation"],
        _S["block_red_scalar"],
        _S["block_pink_position"],
        _S["block_pink_rotation"],
        _S["block_pink_scalar"],
    ],
    SET_SRPB: [
        _S["ee_position"],
        _S["ee_rotation"],
        _S["ee_scalar"],
        _S["base__slide_position"],
        _S["base__slide_rotation"],
        _S["base__slide_scalar"],
        _S["base__drawer_position"],
        _S["base__drawer_rotation"],
        _S["base__drawer_scalar"],
        _S["base__button_position"],
        _S["base__button_rotation"],
        _S["base__button_scalar"],
        _S["led_position"],
        _S["led_rotation"],
        _S["block_red_position"],
        _S["block_red_rotation"],
        _S["block_red_scalar"],
        _S["block_pink_position"],
        _S["block_pink_rotation"],
        _S["block_pink_scalar"],
        _S["block_blue_position"],
        _S["block_blue_rotation"],
        _S["block_blue_scalar"],
    ],
    "bp": [
        _S["ee_position"],
        _S["ee_rotation"],
        _S["ee_scalar"],
        _S["base__slide_position"],
        _S["base__slide_rotation"],
        _S["base__slide_scalar"],
        _S["base__drawer_position"],
        _S["base__drawer_rotation"],
        _S["base__drawer_scalar"],
        _S["base__button_position"],
        _S["base__button_rotation"],
        _S["base__button_scalar"],
        _S["led_position"],
        _S["led_rotation"],
        _S["block_pink_position"],
        _S["block_pink_rotation"],
        _S["block_pink_scalar"],
    ],
    "bb": [
        _S["ee_position"],
        _S["ee_rotation"],
        _S["ee_scalar"],
        _S["base__slide_position"],
        _S["base__slide_rotation"],
        _S["base__slide_scalar"],
        _S["base__drawer_position"],
        _S["base__drawer_rotation"],
        _S["base__drawer_scalar"],
        _S["base__button_position"],
        _S["base__button_rotation"],
        _S["base__button_scalar"],
        _S["led_position"],
        _S["led_rotation"],
        _S["block_blue_position"],
        _S["block_blue_rotation"],
        _S["block_blue_scalar"],
    ],
}
