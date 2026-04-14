from hoopgn.skills.branches.hoopgn.hoopgn_skill import HoopGNSkillConfig
from hoopgn.buffer import BufferConfig

from hoopgn.environments.calvin import CalvinEnvironmentConfig
from hoopgn.environments.environment import EnvironmentConfig
from hoopgn.environments.ogbench import OGBenchEnvironmentConfig
from hoopgn.logger import LogMode, LoggerConfig
from hoopgn.networks.baseline import BaselineNetworkConfig
from hoopgn.networks.v1 import HoopgnV1Config
from hoopgn.networks.network import NetworkConfig


def ppo_default_config(
    network: NetworkConfig,
    buffer: BufferConfig,
    eval: bool,
) -> HoopGNSkillConfig:
    return HoopGNSkillConfig(
        network=network,
        buffer=buffer,
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
        HoopgnV1Config(
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
    elif tag == "ogbench":
        return OGBenchEnvironmentConfig()
    else:
        raise ValueError(f"Unknown environment tag: {tag}")
