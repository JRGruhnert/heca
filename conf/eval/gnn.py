from src.agents.ppo import PPOAgentConfig
from src.environments.calvin import CalvinEnvironmentConfig
from src.buffer import BufferConfig
from src.logger import LogMode, LoggerConfig
from src.storage import StorageConfig
from src.experiments.noise_experiment import NoiseExperimentConfig
from cli.hoopgn import TrainConfig
from conf.common.evaluator import dense3_evaluator
from src.networks.gnn import GraphNetworkConfig

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

config = TrainConfig(
    agent=PPOAgentConfig(
        network=GraphNetworkConfig(),
        eval=eval,
        max_batches=1,
        early_stop_patience=1,
        min_batches=1,
        retrain=retrain,
        use_ema_for_early_stopping=False,
    ),
    buffer=BufferConfig(steps=2048),
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
        p_empty=0.0,
        p_rand=0.0,
    ),
    environment=CalvinEnvironmentConfig(render=render),
    evaluator=dense3_evaluator,
)
