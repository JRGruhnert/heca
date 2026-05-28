from scripts.explain import ExplainConfig
from src.agents.ppo.gnn import GNNAgentConfig
from src.environments.calvin import CalvinEnvironmentConfig
from src.modules.buffer import BufferConfig
from src.modules.logger import LogMode
from src.modules.storage import StorageConfig
from src.experiments.pepr import PePrConfig
from conf.common.evaluator import sparse_evaluator

mode = LogMode.WANDB
render = False
eval = True
network = "gnn"
prefix = "t"

skills_eval_states = "blue"
used_states = "blue"
p_empty = 0.0
p_rand = 0.0

tag = f"{prefix}_{used_states}_{skills_eval_states}"
wandb_tag = f"{network}_{tag}"

config = ExplainConfig(
    agent=GNNAgentConfig(
        eval=eval,
        max_batches=750,
        early_stop_patience=50,
        min_batches=250,
        use_ema_for_early_stopping=False,
    ),
    buffer=BufferConfig(steps=1024),
    storage=StorageConfig(
        used_skills=skills_eval_states,
        used_states=used_states,
        eval_states=skills_eval_states,
        tag=tag,
        network=network,
    ),
    experiment=PePrConfig(
        p_empty=p_empty,
        p_rand=p_rand,
    ),
    environment=CalvinEnvironmentConfig(render=render),
    evaluator=sparse_evaluator,
)
