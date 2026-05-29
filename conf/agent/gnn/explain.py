from scripts.explain import ExplainConfig
from conf.agent.gnn.t_train import t_train_config
from conf.agent.gnn.s_train import s_train_config

config = ExplainConfig(
    source=s_train_config,
    target=t_train_config,
)
