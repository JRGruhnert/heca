from hoopgn.agents.branches.hoopgn_agent import HoopGNSkill
from hoopgn.misc.buffer import BufferConfig

from hoopgn.environments.calvin import CalvinEnvironment
from hoopgn.environments.environment import Environment
from hoopgn.environments.ogbench import OGBenchEnvironment
from hoopgn.misc.logger import LogMode, LoggerConfig
from hoopgn.networks.mp_baseline import MPBaseline
from hoopgn.networks.mp_gnn import MPGnn
from hoopgn.networks.mp_final import MPNetwork


def ppo_default_config(
    network: MPNetwork.Config,
    buffer: BufferConfig,
    eval: bool,
) -> HoopGNSkill.Config:
    return HoopGNSkill.Config(
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
) -> MPNetwork.Config:
    network_name = "gnn" if is_gnn else "baseline"
    checkpoint_path = (
        f"results/{network_name}/{checkpoint_name}/model_cp_best.pt"
        if checkpoint_name
        else None
    )
    return (
        MPGnn.Config(
            environment=CalvinEnvironment.Signature(),
            checkpoint_path=checkpoint_path,
            explain_mode=explain_mode,
        )
        if is_gnn
        else MPBaseline.Config(
            environment=CalvinEnvironment.Signature(),
            checkpoint_path=checkpoint_path,
        )
    )
