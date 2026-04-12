from hoopgn.agents.ppo import PPOAgentConfig
from hoopgn.environments.calvin import CalvinEnvironmentConfig
from hoopgn.buffer import BufferConfig
from hoopgn.logger import LogMode, LoggerConfig
from hoopgn.storage import StorageConfig
from hoopgn.experiments.noise_experiment import NoiseExperimentConfig


mode = LogMode.WANDB
render = False
eval = False
network = "gnn"
prefix = "s"

skills_eval_states = "sr"
used_states = "srpb"
p_empty = 0.0
p_rand = 0.0

tag = f"{prefix}_{used_states}_{skills_eval_states}"
wandb_tag = f"{network}_{tag}"

buffer = BufferConfig(steps=1024)
logger = LoggerConfig(
    mode=mode,
    wandb_tag=wandb_tag,
)
storage = StorageConfig(
    used_skills=skills_eval_states,
    used_states=used_states,
    states_eval=skills_eval_states,
    tag=tag,
    network=network,
)
experiment = NoiseExperimentConfig(
    p_empty=p_empty,
    p_rand=p_rand,
)
environment = CalvinEnvironmentConfig(render=render)
evaluator = sparse_evaluator
