from scripts.explain import ExplainConfig
from conf.agent.gnn.bp_train import bp_config
from conf.agent.gnn.bb_train import bb_config
from conf.agent.gnn.br_train import br_config

config = ExplainConfig(
    bb=bb_config,
    bp=bp_config,
    br=br_config,
)
