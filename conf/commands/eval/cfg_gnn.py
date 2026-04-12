from cli.commands.train import TrainerConfig
from hoopgn.agents.ppo import PPOAgentConfig
from hoopgn.environments.calvin import CalvinEnvironmentConfig
from hoopgn.buffer import BufferConfig
from hoopgn.logger import LogMode, LoggerConfig
from hoopgn.storage import StorageConfig
from hoopgn.experiments.noise_experiment import NoiseExperimentConfig
from hoopgn.networks.gnn import GraphNetworkConfig

mode = LogMode.TERMINAL
render = False
retrain = False
eval = True

network = "gnn"
checkpoint_tag = f"t_blue_blue_pe0.0_pr0.0"

skills_eval_states = "red"
used_states = "red"


prefix = "d_blue"
tag = f"{prefix}_{used_states}_{skills_eval_states}"
wandb_tag = f"{network}_{tag}"

config = TrainerConfig(
    agent=PPOAgentConfig(
        network=GraphNetworkConfig(),
        batch_size=2048,
        eval=eval,
        max_batches=1,
        early_stop_patience=1,
        min_batches=1,
        retrain=retrain,
        use_ema_for_early_stopping=False,
    ),
    logger=LoggerConfig(
        mode=mode,
        wandb_tag=wandb_tag,
    ),
    storage=StorageConfig(
        used_skills=skills_eval_states,
        used_states=used_states,
        states_eval=skills_eval_states,
        tag=tag,
        network=network,
        checkpoint_path=f"results/{network}/{checkpoint_tag}/model_cp_best.pth",
    ),
    experiment=NoiseExperimentConfig(
        environment=CalvinEnvironmentConfig(),
        p_empty=0.0,
        p_rand=0.0,
    ),
)
