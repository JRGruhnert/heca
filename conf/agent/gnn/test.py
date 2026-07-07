from src.agents.ppo.gnn import GNNAgentConfig
from src.environments.calvin import CalvinEnvironmentConfig
from src.modules.buffer import BufferConfig
from src.modules.logger import LogMode, LoggerConfig
from src.modules.storage import StorageConfig
from src.experiments.pepr import PePrConfig
from scripts.train import TrainConfig
from conf.common.evaluator import dense3_evaluator

mode = LogMode.TERMINAL
render = False
retrain = False
eval = True

source_tag = "test"
target_tag = "test"

network = "gnn"
checkpoint_tag = f"t_{source_tag}_{source_tag}_pe0.0_pr0.0"


prefix = "exp"
tag = f"{prefix}_{source_tag}_{target_tag}"
wandb_tag = f"{network}_{tag}"

config = TrainConfig(
    agent=GNNAgentConfig(
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
        used_skills=target_tag,
        used_states=target_tag,
        eval_states=target_tag,
        tag=tag,
        network=network,
        checkpoint_path=f"results/{network}/{checkpoint_tag}/model_cp_best.pth",
    ),
    experiment=PePrConfig(
        p_empty=0.0,
        p_rand=0.0,
    ),
    environment=CalvinEnvironmentConfig(render=render),
    evaluator=dense3_evaluator,
)
